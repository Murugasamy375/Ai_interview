import json
import logging
from typing import List, Dict, Any, Optional
from app.services.llm import call_grok

logger = logging.getLogger("app.services.agents")

async def run_screening_agent(
    resume_text: str,
    jd_text: str,
    skill_gaps: List[Dict[str, Any]],
    ats_score: float,
    api_key: str = ""
) -> Dict[str, Any]:
    """
    Screening Agent: Analyzes resume, JD, gaps, and ATS score.
    Returns a candidate profile with dynamically generated interview topics.
    
    This agent acts as an experienced tech interviewer who:
    - Reads the resume and JD for the first time
    - Identifies strengths and gaps
    - Creates a strategic interview plan
    - Detects if candidate is fresher level
    - Calculates appropriate question count
    """
    system_prompt = """You are an experienced technical interviewer. Your job is to create a strategic interview plan for a candidate based on their resume and the job requirements.

YOU ARE NOT RE-SCORING THE MATCH. The ATS score already exists. Your job is to plan what to ask.

Think like a real tech interviewer:
1. What do I see in their resume that's clearly relevant to this role?
2. What critical skills/experience are they missing or weak in?
3. What would I want to drill into to understand this candidate's actual capabilities?
4. How can I structure the interview to be fair but thorough?
5. Is this candidate a fresher (0-2 years exp) or more experienced? Adjust difficulty accordingly.

You will receive:
- Resume (their actual background and experience)
- Job Description (what this role requires)
- Skill gaps (areas where resume doesn't match JD requirements)
- ATS score (overall match percentage)

FRESHER DETECTION:
- Count total years of experience mentioned in resume
- Fresher = 0-2 years total experience, no senior roles
- Junior = 2-5 years experience
- Senior = 5+ years or leadership/architect roles

QUESTION COUNT RULES:
- Fresher level: Generate 2-3 UNIQUE TOPICS (will be expanded to 6-9 questions: 3 per topic)
- Junior level: Generate 2-3 UNIQUE TOPICS (will be expanded to 6-9 questions: 3 per topic)
- Senior level: Generate 3-4 UNIQUE TOPICS (will be expanded to 9-12 questions: 3 per topic)
- Each topic will automatically be asked 3 times with different angles
- Base it on: number of key skills to verify, role complexity, candidate level

IMPORTANT: Generate topics that directly come from comparing the resume against the JD requirements. Don't use generic topics.

Return ONLY valid JSON in this exact structure, with no preamble or explanation:

{
  "strengths": ["<specific skill/technology from resume that matches JD>", ...],
  "weak_areas": ["<specific gap or missing skill from JD not in resume>", ...],
  "topic_order": [
    "<specific technical topic derived from JD that matches/tests the candidate>",
    "<another specific technical skill or responsibility from JD>",
    ...
  ],
  "candidate_summary": "<2-3 sentence neutral summary of the candidate's background relevant to this role>",
  "suggested_question_count": <integer between 5 and 8 for fresher, higher for experienced>,
  "is_fresher": <boolean: true if 0-2 years experience>,
  "experience_level": "<fresher|junior|senior>"
}

RULES FOR topic_order:
- Each topic MUST be specific and concrete (e.g., "Experience designing database schemas" not "Database knowledge")
- Topics should directly reference things in either the resume or JD
- Mix strengths and gaps so the interview is balanced, not purely adversarial
- Start with areas where you want to confirm strengths
- Then probe the weak areas systematically
- Topics should be answerable questions or skill areas, not vague statements
- Order by importance to the role (not by gap size)
- FOR FRESHER: Avoid asking too many advanced follow-ups, keep foundational

RULES FOR strengths:
- Only include skills/technologies the candidate has concrete evidence for in their resume
- Reference the specific projects or experience that demonstrate the strength
- Include relevant technologies, frameworks, or methodologies

RULES FOR weak_areas:
- List specific skills or technologies required by the JD that are missing or weak in the resume
- Order by criticality to the role success
- Be specific: "No experience with Kubernetes" not "Cloud infrastructure"

EXAMPLE:
If JD requires "5+ years Python development" and resume shows "2 years Python", the weak area is literally: "Limited Python experience (2 years vs 5+ required)"
If JD requires "RESTful API design" and resume mentions "built 3 REST APIs", a strength is: "RESTful API design (3 production APIs in resume)"
"""

    # Format the gaps more descriptively
    simplified_gaps = [
        f"• {gap.get('text', '').strip()[:120]}..."
        for gap in skill_gaps
    ]

    user_prompt = f"""=== CANDIDATE RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{jd_text}

=== SKILL GAPS IDENTIFIED (via semantic matching) ===
{chr(10).join(simplified_gaps) if simplified_gaps else "None identified"}

=== MATCHING METRICS ===
- ATS Match Score: {ats_score}%

=== YOUR TASK ===
As a tech interviewer reviewing this resume against this JD, create your interview strategy:

1. What are 2-3 CONCRETE STRENGTHS you see that match this role?
2. What are 2-4 SPECIFIC GAPS or areas to probe?
3. Generate 2-3 UNIQUE TOPICS to interview on (each topic will be asked 3 times = 6-9 total questions)
4. Summarize the candidate neutrally in 2-3 sentences
5. How many UNIQUE TOPICS? (2-3 for fresher, 3-4 for experienced)
6. Is this candidate a FRESHER (0-2 years exp) or more experienced?

IMPORTANT: topic_order should have UNIQUE topics only. System will expand each to 3 questions per topic.

Generate the screening profile as specified in your instructions."""

    logger.info("🔍 Running Screening Agent - Analyzing resume against JD...")
    profile = await call_grok(system_prompt, user_prompt, api_key=api_key, json_response=True)
    
    # Expand topic_order: Each topic appears 3 times (3 questions per topic)
    original_topics = profile.get('topic_order', [])
    expanded_topics = []
    for topic in original_topics:
        expanded_topics.extend([topic, topic, topic])  # 3 questions per topic
    
    profile['topic_order'] = expanded_topics
    # Update suggested_question_count to reflect total questions (unique topics × 3)
    profile['suggested_question_count'] = len(expanded_topics)
    
    logger.info(f"✅ Screening Agent completed - Generated {len(original_topics)} unique topics × 3 questions = {len(expanded_topics)} total questions")
    return profile

async def run_interviewer_agent(
    profile: Dict[str, Any],
    skill_gaps: List[Dict[str, Any]],
    questions_asked: List[Dict[str, Any]],
    target_topic: str,
    api_key: str = "",
    is_fresher: bool = True,
    skipped_topics: List[str] = None
) -> Dict[str, Any]:
    """
    Interviewer Agent: Generates technical interview questions with human-like thinking.
    
    Acts as an experienced tech interviewer who:
    - Listens to previous answers and builds on them
    - Asks follow-ups based on candidate's responses
    - References specific things they mentioned
    - Adapts strategy based on how well they performed
    - Shows natural conversation flow
    - Scales difficulty based on candidate level (fresher vs experienced)
    - Skips follow-up questions for topics where candidate struggled
    """
    if skipped_topics is None:
        skipped_topics = []

    system_prompt = """You are an experienced technical interviewer conducting a natural, flowing conversation.

YOUR ROLE: You're interviewing a candidate. You've been talking to them, you've heard their answers, and now you're deciding what to ask next.

DIFFICULTY LEVEL: This is a FRESHER-level interview. Keep questions foundational and practical - avoid advanced architectural patterns or edge cases. Focus on core fundamentals, real-world experience, and problem-solving approach.

HUMAN INTERVIEWER THINKING:
1. If they answered well before (score 70+): Dig deeper into that area or move to next topic
2. If they struggled (score <70): Ask a follow-up or foundational question to understand better
3. If this is early in interview: Start with their strengths to build confidence
4. Listen to what they ACTUALLY said: Reference specific things, projects, technologies
5. Build natural conversation: "You mentioned [X], can you tell me more about..." 
6. Don't just jump to new topics: Connect to what they said before
7. For fresher candidates: Keep it real-world and practical, avoid theoretical discussions
8. Show you're listening and interested in their actual experience

FRESHER-LEVEL GUIDELINES:
- Ask about real projects they've worked on (not theory)
- Focus on fundamentals and practical skills
- Avoid advanced topics like microservices architecture, distributed systems internals, advanced design patterns
- Ask "How would you approach?" rather than "What's the best pattern?"
- Probe hands-on experience over academic knowledge
- Build confidence by starting with their strengths

CONVERSATION STRATEGIES:

For FOLLOW-UP questions (when they gave a mediocre answer or you want deeper insight):
- "You mentioned [specific thing they said]. Can you walk me through how you approached [problem]?"
- "In your [project they mentioned], did you have to handle [related challenge]?"
- "That's interesting. How would you handle [similar scenario] in production?"
- "Tell me more about [specific technology/concept they mentioned]"

For STRENGTH AREAS (when they answered well):
- "Great! Since you have experience with [what they said], have you dealt with [common challenge]?"
- "That's a solid approach. Tell me about a time you had to [take it further/scale/optimize]."
- "I see you've used [tech], what would you do if you had to [refactor/migrate/handle edge cases]?"

For SKILL GAPS (areas they're weak in):
- "I see you're new to [area]. How would you approach learning [specific technology]?"
- "What's your experience level with [required skill]? Walk me through what you know."
- "Have you worked with [requirement] before? If so, tell me about the project."

MOST IMPORTANT: 
- Reference their actual answers and projects
- Show you're listening and engaged
- Make it conversational and friendly
- Build on previous answers rather than jumping topics

Return ONLY valid JSON in this exact structure, with no preamble or explanation:
{
  "question": "<the next interview question, building on context>",
  "target_topic": "<the skill/topic being probed>"
}
"""
    
    # Adjust system prompt if not fresher
    if not is_fresher:
        system_prompt = system_prompt.replace(
            "DIFFICULTY LEVEL: This is a FRESHER-level interview. Keep questions foundational and practical - avoid advanced architectural patterns or edge cases. Focus on core fundamentals, real-world experience, and problem-solving approach.",
            "DIFFICULTY LEVEL: This is an EXPERIENCED-level interview. You can ask deeper technical questions about system design, optimization, and architectural thinking."
        ).replace(
            "For fresher candidates: Keep it real-world and practical, avoid theoretical discussions",
            "For experienced candidates: Ask about architectural decisions, trade-offs, and how they've handled complex scenarios at scale"
        )


    # Adjust system prompt if not fresher
    if not is_fresher:
        system_prompt = system_prompt.replace(
            "DIFFICULTY LEVEL: This is a FRESHER-level interview. Keep questions foundational and practical - avoid advanced architectural patterns or edge cases. Focus on core fundamentals, real-world experience, and problem-solving approach.",
            "DIFFICULTY LEVEL: This is an EXPERIENCED-level interview. You can ask deeper technical questions about system design, optimization, and architectural thinking."
        ).replace(
            "For fresher candidates: Keep it real-world and practical, avoid theoretical discussions",
            "For experienced candidates: Ask about architectural decisions, trade-offs, and how they've handled complex scenarios at scale"
        )

    # First question should always be "Tell me about yourself"
    if not questions_asked:
        logger.info("🎯 First question - Starting with standard opening")
        return {
            "question": "Tell me about yourself. Please share your background, experience, and what brings you to this role.",
            "target_topic": "Candidate Background & Motivation"
        }

    # Build detailed context from previous questions and answers
    conversation_context = ""
    if questions_asked:
        conversation_context = "\n=== INTERVIEW CONVERSATION SO FAR ===\n"
        for i, q in enumerate(questions_asked, 1):
            score = q.get("score", 0)
            answer_preview = q.get("answer", "N/A")[:150]
            
            conversation_context += f"\nQ{i} ({q.get('topic', 'N/A')}): {q.get('question', 'N/A')[:100]}...\n"
            conversation_context += f"A{i}: {answer_preview}...\n"
            conversation_context += f"Score: {score}/100\n"
    else:
        conversation_context = "\n=== THIS IS THE FIRST QUESTION ===\nNo previous conversation yet. Start with their strengths to build rapport."

    # Analyze performance pattern
    if questions_asked:
        scores = [q.get("score", 0) for q in questions_asked]
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 75:
            performance_note = f"Candidate is performing WELL (avg {avg_score:.0f}/100) - Dig deeper and challenge them"
        elif avg_score >= 50:
            performance_note = f"Candidate is OKAY (avg {avg_score:.0f}/100) - Mix of follow-ups and new topics"
        else:
            performance_note = f"Candidate is STRUGGLING (avg {avg_score:.0f}/100) - Focus on foundational understanding"
        
        conversation_context += f"\n=== PERFORMANCE SUMMARY ===\n{performance_note}"

    # Get candidate's best answer to reference
    if questions_asked:
        best_answer = max(questions_asked, key=lambda x: x.get("score", 0))
        conversation_context += f"\nBEST ANSWER: Q: '{best_answer.get('question', '')[:60]}...' - They showed: '{best_answer.get('answer', '')[:80]}...'"

    # Count how many questions already asked on this topic
    questions_on_this_topic = sum(1 for q in questions_asked if q.get("topic") == target_topic)
    
    # Format candidate profile
    strengths_list = ", ".join(profile.get("strengths", [])[:3]) if profile.get("strengths") else "TBD"
    weak_areas_list = ", ".join(profile.get("weak_areas", [])[:3]) if profile.get("weak_areas") else "TBD"

    # Build context about how many questions we've already asked on this topic
    topic_depth_note = ""
    if questions_on_this_topic == 0:
        topic_depth_note = f"\nThis is the FIRST question about '{target_topic}' - Start with basics/overview."
    elif questions_on_this_topic == 1:
        topic_depth_note = f"\nThis is the SECOND question about '{target_topic}' - Go DEEPER or ask a different angle (e.g., examples, challenges, trade-offs)."
    elif questions_on_this_topic >= 2:
        topic_depth_note = f"\nThis is the THIRD question about '{target_topic}' - Ask the MOST ADVANCED angle or real-world scenario."

    user_prompt = f"""=== ABOUT THE CANDIDATE ===
Background: {profile.get('candidate_summary', 'N/A')}
Strengths: {strengths_list}
Weak Areas: {weak_areas_list}

{conversation_context}

=== NEXT TOPIC TO EXPLORE ===
Target: {target_topic}
{topic_depth_note}

Related Skills from Job Description:
{json.dumps([g.get('text', '')[:80] for g in skill_gaps[:3]], indent=2)}

=== THINK LIKE A REAL INTERVIEWER ===

1. What did they say in previous answers? (Any specific projects, technologies, approaches?)
2. How are they performing? (Strong? Weak? Improving?)
3. How many times have we asked about this topic? (Adapt depth accordingly)
4. Should this be a deeper dive or different angle?

QUESTION PROGRESSION FOR SAME TOPIC:
- Q1 (first time): "Tell me about your experience with [topic]"
- Q2 (second time): "You mentioned [their answer]. Tell me about [specific challenge/scenario]"
- Q3 (third time): "That's good. How would you handle [advanced scenario] in production?"

Generate ONE natural, conversational question for "{target_topic}".
"""

    logger.info(f"🧠 Running Interviewer Agent - Thinking like a human...")
    logger.info(f"   Topic: '{target_topic}' (Q#{questions_on_this_topic + 1}), Previous answers: {len(questions_asked)}")
    
    question_res = await call_grok(system_prompt, user_prompt, api_key=api_key, json_response=True)
    logger.info(f"✅ Question generated: {question_res.get('question', '')[:80]}...")
    
    return question_res


async def run_evaluator_agent(
    question: str,
    answer: str,
    api_key: str = ""
) -> Dict[str, Any]:
    """
    Evaluator Agent: Scores and evaluates candidate answers like a real tech interviewer.
    
    Considers:
    - Technical accuracy and depth
    - Communication clarity
    - Problem-solving approach
    - Evidence of real experience
    """
    system_prompt = """You are a senior technical interviewer evaluating a candidate's answer to a technical question.

Your job: Score and provide feedback on the answer as if you're actually conducting the interview.

SCORING RUBRIC:
90-100 (Excellent):
- Technically accurate and complete
- Shows deep understanding
- Provides specific examples
- Explains reasoning clearly
- Demonstrates hands-on experience

70-89 (Good):
- Demonstrates solid understanding
- Mostly accurate with minor gaps
- Some examples but could be more specific
- Clear communication
- Shows practical knowledge

50-69 (Adequate):
- Shows basic understanding
- Some inaccuracies or missing concepts
- Vague or generic examples
- Communication could be clearer
- Limited concrete evidence

0-49 (Poor/Weak):
- Significant misunderstandings
- Inaccurate or incomplete answer
- No concrete examples
- Unclear communication
- Suggests limited hands-on experience

EVALUATION FACTORS:
1. Technical Accuracy: Does the answer correctly address the question?
2. Depth: Does it show deep understanding or just surface knowledge?
3. Examples: Are there concrete, specific examples from real work?
4. Clarity: Is the answer well-organized and easy to follow?
5. Experience: Does it demonstrate actual hands-on experience?

BE FAIR but RIGOROUS. This is a professional technical interview.
- Give credit for honest "I don't know" followed by how they'd learn
- Penalize vague answers and generic responses
- Value specific technical details and real examples

Return ONLY valid JSON in this exact structure, with no preamble or explanation:
{
  "score": <integer between 0 and 100>,
  "feedback": "<2-3 sentence constructive feedback - mention strengths and areas for improvement>"
}
"""

    user_prompt = f"""=== INTERVIEW QUESTION ===
{question}

=== CANDIDATE'S ANSWER ===
{answer}

=== EVALUATION TASK ===
Score this answer using the rubric and evaluation factors provided.

Consider:
- Is this answer technically correct?
- Does it show real understanding or superficial knowledge?
- Are there specific examples from real work?
- Is the communication clear?
- What does this tell you about their actual capabilities?

Provide the score and constructive feedback."""

    logger.info("📊 Running Evaluator Agent - Assessing answer quality...")
    evaluation = await call_grok(system_prompt, user_prompt, api_key=api_key, json_response=True)
    score = evaluation.get("score", 0)
    logger.info(f"✅ Answer scored: {score}/100")
    return evaluation

def run_report_agent(
    session_id: str,
    resume_filename: str,
    jd_filename: str,
    ats_score: float,
    candidate_profile: Dict[str, Any],
    questions_asked: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Report Agent: Purely aggregates ATS matching results and individual interview question scores.
    No LLM call needed.
    """
    logger.info("Running Report Agent...")
    scores = [q["score"] for q in questions_asked if q.get("score") is not None]
    overall_interview_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "session_id": session_id,
        "resume_filename": resume_filename,
        "jd_filename": jd_filename,
        "ats_match_score": ats_score,
        "overall_interview_score": round(overall_interview_score, 2),
        "candidate_profile": candidate_profile,
        "questions_asked": [
            {
                "question": q["question"],
                "topic": q.get("topic", ""),
                "answer": q.get("answer", ""),
                "score": q.get("score", 0),
                "feedback": q.get("feedback", "")
            }
            for q in questions_asked
        ]
    }
