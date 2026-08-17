import io
import logging
import os

import numpy as np
import soundfile as sf
from kokoro import KPipeline

logger = logging.getLogger("app.services.text_to_speech")

# Lazy-loaded model
_pipeline = None


def get_kokoro_pipeline():
    """
    Load Kokoro only when TTS is actually requested.
    This reduces startup RAM usage.
    """
    global _pipeline

    if _pipeline is None:
        lang_code = os.getenv("KOKORO_LANG", "a")

        logger.info("Loading Kokoro TTS pipeline...")

        _pipeline = KPipeline(
            lang_code=lang_code
        )

        logger.info("Kokoro loaded successfully")

    return _pipeline


def generate_speech(
    text: str,
    voice: str | None = None,
    speed: float = 1.0
) -> bytes:
    """
    Convert text to WAV audio using Kokoro.

    IMPORTANT:
    This is a synchronous function.
    Do NOT use await generate_speech().
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    pipeline = get_kokoro_pipeline()

    selected_voice = voice or os.getenv(
        "KOKORO_VOICE",
        "af_heart"
    )

    logger.info(
        "Generating speech using voice=%s",
        selected_voice
    )

    audio_chunks = []

    try:
        generator = pipeline(
            text.strip(),
            voice=selected_voice,
            speed=float(speed)
        )

        for _, _, audio in generator:

            if audio is None:
                continue

            # Kokoro may return a PyTorch tensor
            if hasattr(audio, "detach"):
                audio = (
                    audio
                    .detach()
                    .cpu()
                    .numpy()
                )

            audio = np.asarray(audio)

            if audio.size > 0:
                audio_chunks.append(audio)

    except Exception:
        logger.exception("Kokoro speech generation failed")
        raise

    if not audio_chunks:
        raise RuntimeError(
            "Kokoro produced no audio"
        )

    audio = np.concatenate(audio_chunks)

    output = io.BytesIO()

    # Kokoro uses 24 kHz audio
    sf.write(
        output,
        audio,
        24000,
        format="WAV"
    )

    audio_bytes = output.getvalue()

    logger.info(
        "Generated TTS audio: %d bytes",
        len(audio_bytes)
    )

    return audio_bytes