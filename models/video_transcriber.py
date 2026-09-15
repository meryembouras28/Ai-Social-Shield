from pathlib import Path

from faster_whisper import WhisperModel


WHISPER_MODEL_NAME = "tiny"

_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        print("Chargement du modèle Whisper...")

        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1,
        )

        print("Modèle Whisper chargé !")

    return _whisper_model


def transcribe_audio(audio_path):
    if audio_path is None:
        return {
            "text": "",
            "language": None,
            "language_probability": 0.0,
            "segments": [],
        }

    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {
            "text": "",
            "language": None,
            "language_probability": 0.0,
            "segments": [],
        }

    model = get_whisper_model()

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )

    segment_results = []
    text_parts = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        text_parts.append(text)

        segment_results.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
        })

    full_text = " ".join(text_parts).strip()

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4,
        ),
        "segments": segment_results,
    }