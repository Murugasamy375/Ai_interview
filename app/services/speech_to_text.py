import logging
import os
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(
    "app.services.speech_to_text"
)

# Lazy-loaded Whisper model
_model = None


def get_whisper_model():
    """
    Load Whisper only when transcription is requested.
    """

    global _model

    if _model is None:

        model_name = os.getenv(
            "WHISPER_MODEL",
            "tiny"
        )

        device = os.getenv(
            "WHISPER_DEVICE",
            "cpu"
        )

        compute_type = os.getenv(
            "WHISPER_COMPUTE_TYPE",
            "int8"
        )

        logger.info(
            "Loading Whisper model=%s device=%s compute=%s",
            model_name,
            device,
            compute_type
        )

        _model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=2,
            num_workers=1
        )

        logger.info(
            "Whisper loaded successfully"
        )

    return _model


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    language: str = "en"
) -> str:
    """
    Convert recorded audio into text.

    The function is async because the route can await it,
    but the Whisper inference itself is synchronous.
    """

    if not audio_bytes:
        raise ValueError(
            "Audio data is empty"
        )

    suffix = Path(filename).suffix or ".webm"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp:

            temp.write(audio_bytes)
            temp.flush()

            temp_path = temp.name

        model = get_whisper_model()

        segments, info = model.transcribe(
            temp_path,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0
        )

        text_parts = []

        for segment in segments:

            segment_text = segment.text.strip()

            if segment_text:
                text_parts.append(
                    segment_text
                )

        text = " ".join(text_parts).strip()

        logger.info(
            "Transcription completed: %d characters",
            len(text)
        )

        return text

    except Exception:
        logger.exception(
            "Speech-to-text failed"
        )
        raise

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except OSError:
                pass