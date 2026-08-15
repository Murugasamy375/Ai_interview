# ============================================================================
# FastAPI Application Main Entry Point
# AI Interviewer & ATS Matcher Service
# ============================================================================

# Standard Library Imports
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Union, Optional, List

# Third-party Imports
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Local Imports - Services
from app.services.parser import parse_document
from app.services.vector_store import VectorStoreService
from app.services.ats_matcher import compute_ats_score
from app.routes.voice import router as voice_router

from app.services.agents import (
    run_screening_agent,
    run_interviewer_agent,
    run_evaluator_agent,
    run_report_agent
)

# Local Imports - Schemas (Pydantic Models)
from app.schemas.responses import (
    UploadResponse,
    ListDocumentsResponse,
    ResetResponse,
    ATSScoreResponse,
    InterviewSessionState,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    ScreeningProfile,
    QuestionAskedInfo
)

# Local Imports - Configuration
from app.config import GROK_API_KEY

# ============================================================================
# Configuration & Constants
# ============================================================================

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app.main")

# FastAPI Application Configuration
app = FastAPI(
    title="AI Interviewer & Resume ATS Matcher",
    description="Intelligent interview platform combining vector embeddings, "
                "semantic matching, and LLM-powered agents for resume screening and interviewing",
    version="1.0.0",
    contact={
        "name": "Support",
        "description": "Premium AI-powered interview and resume matching service"
    }
)

# Register voice endpoints: /voice/transcribe and /voice/tts
app.include_router(voice_router)

# ============================================================================
# Middleware Setup
# ============================================================================

# Enable CORS (Cross-Origin Resource Sharing) for all origins
# This allows the frontend (index.html) to communicate with the backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# Static Files & UI Setup
# ============================================================================

# Create static directory if it doesn't exist
os.makedirs("app/static", exist_ok=True)

# Mount static files directory for serving frontend (CSS, JS, HTML)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ============================================================================
# Session Management (In-Memory Database)
# ============================================================================

# Dictionary to store active interview sessions
# Key: session_id (UUID), Value: InterviewSessionState (Pydantic model)
sessions: Dict[str, InterviewSessionState] = {}

# ============================================================================
# Service Initialization
# ============================================================================

# Initialize Vector Store Service
# This service handles:
# - Document chunking and embedding using SentenceTransformer
# - Storage and retrieval from ChromaDB
# - Document management (resumes and job descriptions)
logger.info("🚀 Initializing VectorStoreService...")
vector_store = VectorStoreService()
logger.info("✅ VectorStoreService initialized successfully")

# ============================================================================
# ROOT & UI ENDPOINTS
# ============================================================================

@app.get("/", tags=["UI"])
async def root() -> Dict[str, str]:
    """
    Root endpoint - Welcome message with API documentation link.
    
    Returns:
        dict: Welcome message and documentation URL
    """
    return {
        "message": "Welcome to AI Interviewer & Resume ATS Matcher API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/index", tags=["UI"])
async def get_ui():
    """
    Serve the main web UI (index.html).
    
    This endpoint serves the interactive frontend for the interviewer and ATS matcher.
    The frontend communicates with all other endpoints via fetch API calls.
    
    Returns:
        HTML file content
    """
    logger.info("Serving main UI from index.html")
    return FileResponse("app/static/index.html")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _process_document(
    file: UploadFile,
    doc_type: str,
    vector_store: VectorStoreService
) -> UploadResponse:
    """
    Generic document processing helper function.
    
    Handles PDF/TXT parsing, embedding, and storage for both resumes and job descriptions.
    
    Args:
        file: Uploaded file object
        doc_type: Document type ("resume" or "jd")
        vector_store: VectorStoreService instance
        
    Returns:
        UploadResponse: Document metadata and processing status
        
    Raises:
        HTTPException: If file parsing or storage fails
    """
    logger.info(f"Processing {doc_type} file: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse document (extract text from PDF or read TXT)
        parsed_text = parse_document(file.filename, content)
        
        # Validate extracted text
        if not parsed_text.strip():
            raise ValueError(
                f"{doc_type.upper()} file contains no extractable text. "
                "Please ensure the file is not corrupted or empty."
            )
        
        # Store in vector database
        if doc_type.lower() == "resume":
            result = vector_store.add_resume(file.filename, parsed_text)
        elif doc_type.lower() == "jd":
            result = vector_store.add_jd(file.filename, parsed_text)
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        # Build response
        response_msg = f"{doc_type.title()} successfully processed, embedded, and stored in ChromaDB."
        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            type=result["type"],
            chunks_added=result["chunks_added"],
            timestamp=result["timestamp"],
            message=response_msg
        )
        
    except ValueError as e:
        logger.warning(f"Validation error for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Failed to process {doc_type} file '{file.filename}': {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing {doc_type}: {str(e)}"
        )


# ============================================================================
# DOCUMENT UPLOAD ENDPOINTS
# ============================================================================

@app.post(
    "/upload-resume",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Document Upload"]
)
async def upload_resume(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and process a resume file.
    
    Supported formats: PDF, TXT
    
    The resume is parsed, chunked, embedded using SentenceTransformer,
    and stored in ChromaDB for later retrieval and matching.
    
    Args:
        file: Resume file (PDF or TXT format)
        
    Returns:
        UploadResponse: Document ID, filename, chunks added, and status message
        
    Example:
        POST /upload-resume
        Body: multipart/form-data with file field
    """
    return await _process_document(file, "resume", vector_store)


@app.post(
    "/upload-jd",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Document Upload"]
)
async def upload_jd(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and process a Job Description (JD) file.
    
    Supported formats: PDF, TXT
    
    The JD is parsed, chunked, embedded using SentenceTransformer,
    and stored in ChromaDB for later retrieval and matching.
    
    Args:
        file: Job description file (PDF or TXT format)
        
    Returns:
        UploadResponse: Document ID, filename, chunks added, and status message
        
    Example:
        POST /upload-jd
        Body: multipart/form-data with file field
    """
    return await _process_document(file, "jd", vector_store)



# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get(
    "/documents",
    response_model=ListDocumentsResponse,
    tags=["Document Management"]
)
async def list_documents() -> ListDocumentsResponse:
    """
    Retrieve all stored documents (resumes and job descriptions).
    
    Returns a list of all parsed documents with their metadata including:
    - Document ID (UUID)
    - Filename
    - Document type
    - Number of chunks
    - Upload timestamp
    - Sample text preview
    
    Returns:
        ListDocumentsResponse: Lists of resume and JD metadata
    """
    logger.info("Fetching list of all documents")
    try:
        docs = vector_store.list_documents()
        return ListDocumentsResponse(
            resumes=docs["resumes"],
            job_descriptions=docs["job_descriptions"]
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


@app.post(
    "/reset",
    response_model=ResetResponse,
    tags=["Database Management"]
)
async def reset_database() -> ResetResponse:
    """
    Clear all data from the database.
    
    ⚠️ WARNING: This action is irreversible. All stored resumes, job descriptions,
    and embeddings will be permanently deleted.
    
    Returns:
        ResetResponse: Confirmation message
    """
    logger.warning("🔄 Database reset requested - clearing all documents")
    try:
        vector_store.reset_db()
        return ResetResponse(
            status="success",
            message="✅ All documents have been cleared from the database."
        )
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset database: {str(e)}"
        )


# ============================================================================
# HELPER FUNCTIONS FOR DOCUMENT RETRIEVAL & PROCESSING
# ============================================================================

def _resolve_document_id(
    doc_id_query: str,
    collection_type: str,
    vector_store: VectorStoreService
) -> Optional[str]:
    """
    Resolve a document identifier to its actual UUID.
    
    Supports multiple query formats:
    - UUID string: Direct match against document_id
    - Filename substring: Fuzzy match against filename
    - Index: 0-based or 1-based chronological index
    
    Args:
        doc_id_query: The query identifier
        collection_type: "resumes" or "job_descriptions"
        vector_store: VectorStoreService instance
        
    Returns:
        Resolved document ID (UUID) or None if not found
    """
    docs_info = vector_store.list_documents()
    docs = docs_info.get("resumes", []) if collection_type == "resumes" else docs_info.get("job_descriptions", [])
    
    # Direct UUID match
    for doc in docs:
        if doc["document_id"] == doc_id_query:
            return doc["document_id"]
    
    # Filename substring match
    for doc in docs:
        if doc_id_query.lower() in doc["filename"].lower():
            return doc["document_id"]
    
    # Index-based match (0-based)
    try:
        idx = int(doc_id_query)
        if 0 <= idx < len(docs):
            return docs[idx]["document_id"]
    except ValueError:
        pass
    
    return None


def _extract_topic_string(topic_entry: Union[str, Dict, Any]) -> str:
    """
    Extract topic string from various formats returned by agents.
    
    Topics can be strings, dictionaries, or objects with a topic attribute.
    
    Args:
        topic_entry: Topic entry in various possible formats
        
    Returns:
        Extracted topic as string
    """
    if isinstance(topic_entry, str):
        return topic_entry
    elif isinstance(topic_entry, dict):
        return topic_entry.get("topic", "")
    elif hasattr(topic_entry, "topic"):
        return topic_entry.topic
    else:
        return str(topic_entry)


# ============================================================================
# ATS SCORE & MATCHING ENDPOINTS
# ============================================================================

@app.get(
    "/ats-score",
    response_model=ATSScoreResponse,
    tags=["Resume Matching"]
)
async def get_ats_score(
    resume_id: str,
    jd_id: str
) -> ATSScoreResponse:
    """
    Compute hybrid ATS (Applicant Tracking System) matching score.
    
    Combines semantic similarity (embeddings) and BM25 keyword matching
    to provide a comprehensive resume-to-job matching score.
    
    Scoring breakdown:
    - Semantic Score: Vector similarity using embeddings (0-1)
    - BM25 Score: Keyword frequency-based ranking
    - ATS Score: Weighted hybrid combination
    
    Args:
        resume_id: Resume identifier (UUID, filename, or index)
        jd_id: Job Description identifier (UUID, filename, or index)
        
    Returns:
        ATSScoreResponse: Detailed scoring breakdown and chunk matches
        
    Raises:
        HTTPException 404: If resume or JD not found
        HTTPException 500: If score computation fails
        
    Example:
        GET /ats-score?resume_id=john_resume.pdf&jd_id=python_developer_jd.pdf
    """
    logger.info(f"Computing ATS score for resume_id={resume_id}, jd_id={jd_id}")
    try:
        # Retrieve resume chunks
        resume_chunks = vector_store.get_document_chunks("resumes", resume_id)
        if not resume_chunks:
            logger.warning(f"Resume not found: {resume_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume matching '{resume_id}' was not found in database."
            )

        # Retrieve JD chunks
        jd_chunks = vector_store.get_document_chunks("job_descriptions", jd_id)
        if not jd_chunks:
            logger.warning(f"Job description not found: {jd_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job Description matching '{jd_id}' was not found in database."
            )

        # Resolve and get metadata
        docs_info = vector_store.list_documents()
        
        resume_doc_id = resume_chunks[0]["chunk_id"].split("_chunk_")[0]
        resume_doc = next(
            (d for d in docs_info["resumes"] if d["document_id"] == resume_doc_id),
            None
        )
        resume_filename = resume_doc["filename"] if resume_doc else "Unknown Resume"

        jd_doc_id = jd_chunks[0]["chunk_id"].split("_chunk_")[0]
        jd_doc = next(
            (d for d in docs_info["job_descriptions"] if d["document_id"] == jd_doc_id),
            None
        )
        jd_filename = jd_doc["filename"] if jd_doc else "Unknown Job Description"

        # Compute hybrid ATS scores
        score_result = compute_ats_score(resume_chunks, jd_chunks)

        logger.info(
            f"ATS score computed: {score_result['ats_score']:.2f} "
            f"(semantic: {score_result['semantic_score']:.2f}, "
            f"bm25: {score_result['bm25_score']:.2f})"
        )

        return ATSScoreResponse(
            resume_id=resume_doc_id,
            resume_filename=resume_filename,
            jd_id=jd_doc_id,
            jd_filename=jd_filename,
            semantic_score=score_result["semantic_score"],
            bm25_score=score_result["bm25_score"],
            ats_score=score_result["ats_score"],
            chunk_matches=score_result["chunk_matches"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compute ATS score: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute ATS score: {str(e)}"
        )


# ============================================================================
# INTERVIEW SESSION ENDPOINTS
# ============================================================================

@app.post(
    "/start-interview",
    response_model=InterviewSessionState,
    status_code=status.HTTP_201_CREATED,
    tags=["Interview"]
)
async def start_interview(
    resume_id: str,
    jd_id: str,
    x_grok_api_key: Optional[str] = Header(None, alias="X-Grok-API-Key")
) -> InterviewSessionState:
    """
    Initiate a new adaptive interview session.
    
    This endpoint creates an interview session by:
    1. Computing ATS score and identifying skill gaps
    2. Running Screening Agent to create candidate profile
    3. Generating the first interview question
    4. Storing session state for subsequent interactions
    
    The Screening Agent analyzes:
    - Resume strengths and weak areas
    - Skill gaps vs job requirements
    - Optimal question topic order
    - Expected interview duration
    
    Args:
        resume_id: Resume identifier (UUID, filename, or index)
        jd_id: Job Description identifier (UUID, filename, or index)
        x_grok_api_key: Optional Groq API key for LLM calls (header: X-Grok-API-Key)
        
    Returns:
        InterviewSessionState: Created session with first question and metadata
        
    Raises:
        HTTPException 404: If resume or JD not found
        HTTPException 500: If session creation fails
        
    Example:
        POST /start-interview?resume_id=john_resume.pdf&jd_id=python_dev_jd.pdf
        Headers: X-Grok-API-Key: gsk_...
    """
    logger.info(f"Starting interview for resume={resume_id}, jd={jd_id}")
    try:
        # ─────────────────────────────────────────────────────────────────
        # Step 1: Retrieve and validate documents
        # ─────────────────────────────────────────────────────────────────
        resume_chunks = vector_store.get_document_chunks("resumes", resume_id)
        if not resume_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume matching '{resume_id}' was not found."
            )

        jd_chunks = vector_store.get_document_chunks("job_descriptions", jd_id)
        if not jd_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job Description matching '{jd_id}' was not found."
            )

        # Resolve document metadata
        docs_info = vector_store.list_documents()
        
        resume_doc_id = resume_chunks[0]["chunk_id"].split("_chunk_")[0]
        resume_doc = next(
            (d for d in docs_info["resumes"] if d["document_id"] == resume_doc_id),
            None
        )
        resume_filename = resume_doc["filename"] if resume_doc else "Unknown Resume"

        jd_doc_id = jd_chunks[0]["chunk_id"].split("_chunk_")[0]
        jd_doc = next(
            (d for d in docs_info["job_descriptions"] if d["document_id"] == jd_doc_id),
            None
        )
        jd_filename = jd_doc["filename"] if jd_doc else "Unknown Job Description"

        # ─────────────────────────────────────────────────────────────────
        # Step 2: Reconstruct full document texts
        # ─────────────────────────────────────────────────────────────────
        resume_text = "\n".join([c["text"] for c in resume_chunks])
        jd_text = "\n".join([c["text"] for c in jd_chunks])

        # ─────────────────────────────────────────────────────────────────
        # Step 3: Compute ATS score and extract skill gaps
        # ─────────────────────────────────────────────────────────────────
        logger.info("Computing ATS score and skill gaps...")
        ats_score_result = compute_ats_score(resume_chunks, jd_chunks)
        ats_score = ats_score_result["ats_score"]
        
        # Extract skill gaps from chunk matches
        skill_gaps = [
            {
                "text": match["jd_chunk_text"],
                "score": match["hybrid_score"]
            }
            for match in ats_score_result["chunk_matches"]
        ]
        logger.info(f"Identified {len(skill_gaps)} skill gaps")

        # ─────────────────────────────────────────────────────────────────
        # Step 4: Run Screening Agent to create candidate profile
        # ─────────────────────────────────────────────────────────────────
        logger.info("Running Screening Agent to create candidate profile...")
        # Use provided API key or fall back to environment variable
        api_key = x_grok_api_key or GROK_API_KEY
        
        screening_profile_data = await run_screening_agent(
            resume_text=resume_text,
            jd_text=jd_text,
            skill_gaps=skill_gaps,
            ats_score=ats_score,
            api_key=api_key
        )
        
        screening_profile = ScreeningProfile(**screening_profile_data)
        logger.info(f"Profile created with {len(screening_profile.topic_order)} topics")

        # ─────────────────────────────────────────────────────────────────
        # Step 5: Generate first interview question
        # ─────────────────────────────────────────────────────────────────
        topic_order = screening_profile.topic_order
        if not topic_order:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Screening agent did not generate any topics for probing."
            )
        
        first_topic = _extract_topic_string(topic_order[0])
        logger.info(f"Generating first question for topic: {first_topic}")

        question_result = await run_interviewer_agent(
            profile=screening_profile.dict(),
            skill_gaps=skill_gaps,
            questions_asked=[],
            target_topic=first_topic,
            api_key=api_key,
            is_fresher=screening_profile.is_fresher,
            skipped_topics=[]
        )
        
        first_question = question_result.get("question", "")
        first_question_topic = question_result.get("target_topic", first_topic)

        # ─────────────────────────────────────────────────────────────────
        # Step 6: Create and store session state
        # ─────────────────────────────────────────────────────────────────
        session_id = str(uuid.uuid4())
        session_state = InterviewSessionState(
            session_id=session_id,
            resume_id=resume_doc_id,
            resume_filename=resume_filename,
            jd_id=jd_doc_id,
            jd_filename=jd_filename,
            ats_score=ats_score,
            status="ongoing",
            candidate_profile=screening_profile,
            questions_asked=[],
            current_question=first_question,
            current_topic=first_question_topic,
            max_questions=screening_profile.suggested_question_count,
            started_at=datetime.now(),
            skipped_topics=[],
            is_fresher=screening_profile.is_fresher
        )

        sessions[session_id] = session_state
        logger.info(f"✅ Interview session created: {session_id}")
        
        return session_state

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start interview session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start interview: {str(e)}"
        )


@app.post(
    "/interview/{session_id}/answer",
    response_model=SubmitAnswerResponse,
    tags=["Interview"]
)
async def submit_answer(
    session_id: str,
    payload: SubmitAnswerRequest,
    x_grok_api_key: Optional[str] = Header(None, alias="X-Grok-API-Key")
) -> SubmitAnswerResponse:
    """
    Submit candidate's answer to current interview question.
    
    Workflow:
    1. Evaluates candidate's answer using Evaluator Agent
    2. Scores answer and provides feedback
    3. Updates session history
    4. Either generates next question or completes interview
    
    Answer Evaluation:
    - Uses Evaluator Agent LLM to assess quality
    - Scores on relevance, depth, and technical accuracy
    - Provides personalized feedback for improvement
    
    Args:
        session_id: Interview session ID (UUID)
        payload: SubmitAnswerRequest containing candidate's answer
        x_grok_api_key: Optional Groq API key for LLM calls (header: X-Grok-API-Key)
        
    Returns:
        SubmitAnswerResponse: Evaluation score, feedback, and next question/report
        
    Raises:
        HTTPException 404: If session not found
        HTTPException 400: If session already completed
        HTTPException 500: If answer processing fails
        
    Example:
        POST /interview/{session_id}/answer
        Body: {"answer": "I have 5 years of Python experience..."}
    """
    logger.info(f"Processing answer for session={session_id}")
    
    # ─────────────────────────────────────────────────────────────────
    # Validation: Check if session exists and is ongoing
    # ─────────────────────────────────────────────────────────────────
    if session_id not in sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interview session '{session_id}' not found."
        )

    session = sessions[session_id]
    
    if session.status == "completed":
        logger.warning(f"Attempted to answer in completed session: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This interview session is already completed."
        )

    try:
        # ─────────────────────────────────────────────────────────────────
        # Step 0: Check for end-interview intent
        # ─────────────────────────────────────────────────────────────────
        candidate_answer = payload.answer
        answer_lower = candidate_answer.lower()
        
        # Detect end-interview phrases
        end_interview_phrases = [
            "should i end", "should we end", "can we end", "can i end",
            "shall i end", "shall we end", "i'm done", "im done",
            "that's all", "thats all", "i think we're done", "i think we're good",
            "end the interview", "finish the interview", "no more questions"
        ]
        
        wants_to_end = any(phrase in answer_lower for phrase in end_interview_phrases)
        
        if wants_to_end:
            logger.info(f"Candidate requested interview end: {session_id}")
            session.status = "completed"
            session.current_question = None
            session.current_topic = None

            report = run_report_agent(
                session_id=session.session_id,
                resume_filename=session.resume_filename,
                jd_filename=session.jd_filename,
                ats_score=session.ats_score,
                candidate_profile=session.candidate_profile.dict() if session.candidate_profile else {},
                questions_asked=[q.dict() for q in session.questions_asked]
            )

            logger.info(f"✅ Interview ended by candidate request: {session_id}")
            
            return SubmitAnswerResponse(
                session_status="completed",
                score=0,
                feedback="Interview concluded per your request. Thank you!",
                next_question=None,
                final_report=report
            )

        # ─────────────────────────────────────────────────────────────────
        # Step 1: Check 30-minute interview time limit
        # ─────────────────────────────────────────────────────────────────
        if session.started_at:
            elapsed_minutes = (datetime.now() - session.started_at).total_seconds() / 60
            if elapsed_minutes > 30:
                logger.info(f"30-minute interview limit reached: {session_id}")
                session.status = "completed"
                session.current_question = None
                session.current_topic = None

                report = run_report_agent(
                    session_id=session.session_id,
                    resume_filename=session.resume_filename,
                    jd_filename=session.jd_filename,
                    ats_score=session.ats_score,
                    candidate_profile=session.candidate_profile.dict() if session.candidate_profile else {},
                    questions_asked=[q.dict() for q in session.questions_asked]
                )

                logger.info(f"✅ Interview time limit reached: {session_id}")
                
                return SubmitAnswerResponse(
                    session_status="completed",
                    score=0,
                    feedback="Interview time limit (30 minutes) reached. Thank you!",
                    next_question=None,
                    final_report=report
                )

        # ─────────────────────────────────────────────────────────────────
        # Step 2: Evaluate candidate's answer
        # ─────────────────────────────────────────────────────────────────
        current_question = session.current_question
        current_topic = session.current_topic

        logger.info(f"Evaluating answer for topic: {current_topic}")
        
        # Use provided API key or fall back to environment variable
        api_key = x_grok_api_key or GROK_API_KEY
        
        eval_result = await run_evaluator_agent(
            question=current_question,
            answer=candidate_answer,
            api_key=api_key
        )

        score = eval_result.get("score", 0)
        feedback = eval_result.get("feedback", "")
        logger.info(f"Answer evaluated: score={score}/100")

        # ─────────────────────────────────────────────────────────────────
        # Step 3: Track poor answers - add to skipped_topics if score < 50
        # ─────────────────────────────────────────────────────────────────
        if score < 50 and current_topic not in session.skipped_topics:
            session.skipped_topics.append(current_topic)
            logger.info(f"Added '{current_topic}' to skipped topics (score: {score})")

        # ─────────────────────────────────────────────────────────────────
        # Step 4: Record question and answer in session history
        # ─────────────────────────────────────────────────────────────────
        question_record = QuestionAskedInfo(
            question=current_question,
            topic=current_topic,
            answer=candidate_answer,
            score=score,
            feedback=feedback
        )
        session.questions_asked.append(question_record)
        logger.info(f"Question recorded. Total questions: {len(session.questions_asked)}")

        # ─────────────────────────────────────────────────────────────────
        # Step 5: Determine if interview should continue or conclude
        # ─────────────────────────────────────────────────────────────────
        num_asked = len(session.questions_asked)
        profile = session.candidate_profile
        topic_order = profile.topic_order if profile else []

        # Find remaining topics that haven't been skipped
        remaining_topics = [t for t in topic_order[num_asked:] if _extract_topic_string(t) not in session.skipped_topics]
        
        # Interview ends only if:
        # 1. All topics are exhausted/skipped, OR
        # 2. 30+ minutes passed AND no more topics remain
        no_more_topics = num_asked >= len(topic_order) or len(remaining_topics) == 0
        time_limit_reached = elapsed_minutes > 30
        
        should_end_interview = no_more_topics  # Only end when topics are done, regardless of time
        
        if should_end_interview:
            # ───────────────────────────────────────────────────────────
            # INTERVIEW CONCLUSION: Generate final report
            # ───────────────────────────────────────────────────────────
            logger.info(f"Interview complete. Questions: {num_asked}, Time: {elapsed_minutes:.1f} mins, Skipped: {session.skipped_topics}")
            session.status = "completed"
            session.current_question = None
            session.current_topic = None

            report = run_report_agent(
                session_id=session.session_id,
                resume_filename=session.resume_filename,
                jd_filename=session.jd_filename,
                ats_score=session.ats_score,
                candidate_profile=profile.dict() if profile else {},
                questions_asked=[q.dict() for q in session.questions_asked]
            )

            logger.info(f"✅ Interview completed: {session_id}")
            
            return SubmitAnswerResponse(
                session_status="completed",
                score=score,
                feedback=feedback,
                next_question=None,
                final_report=report
            )
        else:
            # ───────────────────────────────────────────────────────────
            # INTERVIEW CONTINUES: Find next question (skip poor-answer topics)
            # ───────────────────────────────────────────────────────────
            
            # Find next topic that's not in skipped_topics
            next_topic = None
            next_topic_index = num_asked
            
            for i in range(num_asked, len(topic_order)):
                candidate_topic = _extract_topic_string(topic_order[i])
                if candidate_topic not in session.skipped_topics:
                    next_topic = candidate_topic
                    next_topic_index = i
                    break
            
            # If all remaining topics were skipped, interview ends
            if next_topic is None:
                logger.info(f"All remaining topics skipped: {session_id}")
                session.status = "completed"
                session.current_question = None
                session.current_topic = None

                report = run_report_agent(
                    session_id=session.session_id,
                    resume_filename=session.resume_filename,
                    jd_filename=session.jd_filename,
                    ats_score=session.ats_score,
                    candidate_profile=profile.dict() if profile else {},
                    questions_asked=[q.dict() for q in session.questions_asked]
                )

                logger.info(f"✅ Interview completed (all topics covered): {session_id}")
                
                return SubmitAnswerResponse(
                    session_status="completed",
                    score=score,
                    feedback=feedback,
                    next_question=None,
                    final_report=report
                )
            
            logger.info(f"Generating next question for topic: {next_topic}")

            # Reconstruct skill gaps for context
            resume_chunks = vector_store.get_document_chunks("resumes", session.resume_id)
            jd_chunks = vector_store.get_document_chunks("job_descriptions", session.jd_id)
            ats_score_result = compute_ats_score(resume_chunks, jd_chunks)
            skill_gaps = [
                {
                    "text": match["jd_chunk_text"],
                    "score": match["hybrid_score"]
                }
                for match in ats_score_result["chunk_matches"]
            ]

            question_result = await run_interviewer_agent(
                profile=profile.dict() if profile else {},
                skill_gaps=skill_gaps,
                questions_asked=[q.dict() for q in session.questions_asked],
                target_topic=next_topic,
                api_key=api_key,
                is_fresher=session.is_fresher,
                skipped_topics=session.skipped_topics
            )

            next_question = question_result.get("question", "")
            next_question_topic = question_result.get("target_topic", next_topic)

            # Update session with next question
            session.current_question = next_question
            session.current_topic = next_question_topic
            
            logger.info(f"Question {num_asked + 1} generated (Time: {elapsed_minutes:.1f} mins)")

            return SubmitAnswerResponse(
                session_status="ongoing",
                score=score,
                feedback=feedback,
                next_question=next_question,
                final_report=None
            )

    except Exception as e:
        logger.error(f"Failed to process answer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process answer: {str(e)}"
        )


@app.get(
    "/interview/{session_id}",
    response_model=InterviewSessionState,
    tags=["Interview"]
)
async def get_interview_session(session_id: str) -> InterviewSessionState:
    """
    Retrieve the current state and history of an interview session.
    
    Returns complete session information including:
    - Session metadata (resume, JD, ATS score)
    - Candidate profile (strengths, weak areas, topic order)
    - Questions asked and answers given
    - Current question being asked
    - Session status (ongoing or completed)
    
    Args:
        session_id: Interview session ID (UUID)
        
    Returns:
        InterviewSessionState: Complete session state
        
    Raises:
        HTTPException 404: If session not found
        
    Example:
        GET /interview/{session_id}
    """
    logger.info(f"Retrieving session state for session={session_id}")
    
    if session_id not in sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interview session '{session_id}' not found."
        )
    
    return sessions[session_id]



# ============================================================================
# Application Ready
# ============================================================================
# The application is now fully configured and ready to handle requests.
# Start with: uvicorn app.main:app --reload
