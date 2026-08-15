"""
Local Text-to-Speech service using Kokoro-82M.

Text flow:
AI question text -> Kokoro -> WAV bytes -> browser

No OpenAI API is used here.
"""

import io
import logging
import os
from typing import Optional

import numpy as np
import soundfile as sf
from kokoro import KPipeline

logger = logging.getLogger("app.services.text_to_speech")

_PIPELINE: Optional[KPipeline] = None


def _get_pipeline() -> KPipeline:
    """Lazy-load Kokoro so the server does not synthesize during import."""
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    # 'a' is the American English pipeline used by Kokoro's examples.
    lang_code = os.getenv("KOKORO_LANG", "a")

    logger.info("Loading Kokoro TTS pipeline lang_code=%s", lang_code)

    _PIPELINE = KPipeline(lang_code=lang_code)

    logger.info("Kokoro TTS pipeline loaded successfully.")
    return _PIPELINE


def _normalise_voice(voice: Optional[str]) -> str:
    """Pick a Kokoro voice."""
    return voice or os.getenv("KOKORO_VOICE", "af_heart")


async def generate_speech(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
) -> bytes:
    """
    Generate WAV audio bytes from text.

    Kokoro's standard sample rate is 24 kHz.
    """

    text = (text or "").strip()

    if not text:
        raise ValueError("TTS text cannot be empty.")

    # Keep individual browser requests reasonably small.
    if len(text) > 4000:
        text = text[:3997] + "..."

    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0

    speed = max(0.5, min(2.0, speed))
    selected_voice = _normalise_voice(voice)

    pipeline = _get_pipeline()

    logger.info(
        "Generating Kokoro speech: voice=%s speed=%.2f chars=%d",
        selected_voice,
        speed,
        len(text),
    )

    audio_parts = []

    # Kokoro may yield multiple chunks for longer text.
    generator = pipeline(
        text,
        voice=selected_voice,
        speed=speed,
    )

    for _, _, audio in generator:
        if audio is None:
            continue

        # Torch tensors -> numpy.
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        else:
            audio = np.asarray(audio)

        audio_parts.append(audio.astype(np.float32))

    if not audio_parts:
        raise RuntimeError("Kokoro returned no audio.")

    audio = np.concatenate(audio_parts)

    output = io.BytesIO()

    # WAV is universally playable by modern browsers and avoids MP3 encoders.
    sf.write(
        output,
        audio,
        24000,
        format="WAV",
        subtype="PCM_16",
    )

    audio_bytes = output.getvalue()

    if not audio_bytes:
        raise RuntimeError("Kokoro generated empty audio.")

    logger.info("Kokoro TTS complete: %d bytes", len(audio_bytes))

    return audio_bytes
