import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile
)
from fastapi.responses import Response
from pydantic import BaseModel

from app.services.speech_to_text import transcribe_audio
from app.services.text_to_speech import generate_speech

logger = logging.getLogger("app.routes.voice")

router = APIRouter(
    prefix="/voice",
    tags=["Voice"]
)


# -----------------------------
# TTS Request
# -----------------------------

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


# -----------------------------
# TEXT → SPEECH
# -----------------------------

@router.post("/tts")
async def tts_endpoint(request: TTSRequest):

    try:

        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )

        logger.info(
            "TTS request received: %s",
            request.text[:100]
        )

        # IMPORTANT:
        # generate_speech() is synchronous.
        # DO NOT use await here.
        audio_bytes = generate_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )

        logger.info(
            "TTS generated successfully: %d bytes",
            len(audio_bytes)
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition":
                    "inline; filename=ai_response.wav"
            }
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "TTS failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# SPEECH → TEXT
# -----------------------------

@router.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default="en")
):

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Audio filename is missing"
            )

        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty"
            )

        logger.info(
            "Transcribing audio: %s",
            file.filename
        )

        text = await transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language
        )

        return {
            "success": True,
            "text": text
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Transcription failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )