"""
Local Speech-to-Text service using faster-whisper.

Audio flow:
Browser microphone -> FastAPI -> faster-whisper -> plain text

No OpenAI API is used here.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger("app.services.speech_to_text")

_MODEL: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Lazy-load the Whisper model so FastAPI can start without loading it immediately."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model_size = os.getenv("WHISPER_MODEL", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv(
        "WHISPER_COMPUTE_TYPE",
        "int8" if device == "cpu" else "float16",
    )

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        model_size,
        device,
        compute_type,
    )

    _MODEL = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    logger.info("faster-whisper model loaded successfully.")
    return _MODEL


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "candidate.webm",
    language: str = "en",
) -> str:
    """Transcribe browser-recorded audio into text."""

    if not audio_bytes:
        raise ValueError("Uploaded audio is empty.")

    suffix = Path(filename).suffix.lower() or ".webm"

    # faster-whisper accepts a path-like audio source. MediaRecorder webm
    # is written temporarily and removed after transcription.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        model = _get_model()

        kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }

        if language:
            kwargs["language"] = language

        logger.info(
            "Transcribing %s (%d bytes)",
            filename,
            len(audio_bytes),
        )

        segments, info = model.transcribe(temp_path, **kwargs)

        # `segments` is lazy; iterating it actually performs inference.
        transcript_parts = [
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ]

        transcript = " ".join(transcript_parts).strip()

        if not transcript:
            raise RuntimeError(
                "No speech was detected. Please speak clearly and try again."
            )

        logger.info(
            "STT complete. language=%s probability=%.3f chars=%d",
            getattr(info, "language", "unknown"),
            getattr(info, "language_probability", 0.0),
            len(transcript),
        )

        return transcript

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temporary audio file: %s", temp_path)
