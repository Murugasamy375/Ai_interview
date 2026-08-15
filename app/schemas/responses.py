from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    type: str
    chunks_added: int
    timestamp: str
    message: str

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    type: str
    chunks_count: int
    upload_timestamp: str
    sample_text: str

class ListDocumentsResponse(BaseModel):
    resumes: List[DocumentMetadata]
    job_descriptions: List[DocumentMetadata]

class ResetResponse(BaseModel):
    status: str
    message: str

class ChunkMatchInfo(BaseModel):
    jd_chunk_index: int
    jd_chunk_text: str
    best_resume_chunk_text: str
    semantic_similarity: float
    bm25_score: float
    hybrid_score: float

class ATSScoreResponse(BaseModel):
    resume_id: str
    resume_filename: str
    jd_id: str
    jd_filename: str
    semantic_score: float
    bm25_score: float
    ats_score: float
    chunk_matches: List[ChunkMatchInfo]

class TopicOrderEntry(BaseModel):
    topic: str
    skill_gap_index: Optional[int] = None
    reason: Optional[str] = None

class ScreeningProfile(BaseModel):
    strengths: List[str]
    weak_areas: List[str]
    topic_order: List[Union[str, TopicOrderEntry, Dict[str, Any]]]
    candidate_summary: str
    suggested_question_count: int
    is_fresher: bool = True  # True if candidate is fresher (0-2 years exp)
    experience_level: str = "fresher"  # fresher, junior, senior

class QuestionAskedInfo(BaseModel):
    question: str
    topic: str
    answer: Optional[str] = None
    score: Optional[int] = None
    feedback: Optional[str] = None

class InterviewSessionState(BaseModel):
    session_id: str
    resume_id: str
    resume_filename: str
    jd_id: str
    jd_filename: str
    ats_score: float
    status: str
    candidate_profile: Optional[ScreeningProfile] = None
    questions_asked: List[QuestionAskedInfo]
    current_question: Optional[str] = None
    current_topic: Optional[str] = None
    max_questions: int
    started_at: datetime = None  # Interview start timestamp
    skipped_topics: List[str] = []  # Topics with poor answers - skip follow-ups
    is_fresher: bool = True  # Whether candidate is fresher level

class SubmitAnswerRequest(BaseModel):
    answer: str

class SubmitAnswerResponse(BaseModel):
    session_status: str
    score: int
    feedback: str
    next_question: Optional[str] = None
    final_report: Optional[Dict[str, Any]] = None
