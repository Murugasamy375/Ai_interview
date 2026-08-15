"""
Voice API routes.

POST /voice/transcribe
    browser audio -> faster-whisper -> transcript

POST /voice/tts
    question text -> Kokoro -> WAV audio
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.speech_to_text import transcribe_audio
from app.services.text_to_speech import generate_speech

logger = logging.getLogger("app.routes.voice")

router = APIRouter(
    prefix="/voice",
    tags=["Voice Interview"],
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@router.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
):
    """Convert candidate microphone audio to text."""

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio is empty.",
            )

        transcript = await transcribe_audio(
            audio_bytes=audio_bytes,
            filename=audio.filename or "candidate.webm",
            language=language,
        )

        return {
            "success": True,
            "text": transcript,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Local STT failed")
        raise HTTPException(
            status_code=500,
            detail=f"Speech-to-text failed: {exc}",
        )


@router.post("/tts")
async def tts_endpoint(payload: TTSRequest):
    """Convert AI interview question text into WAV audio."""

    try:
        audio_bytes = await generate_speech(
            text=payload.text,
            voice=payload.voice,
            speed=payload.speed,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
            },
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Local TTS failed")
        raise HTTPException(
            status_code=500,
            detail=f"Text-to-speech failed: {exc}",
        )
