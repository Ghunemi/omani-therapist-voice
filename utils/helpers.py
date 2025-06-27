import tempfile
import uuid
from langdetect import detect
import whisper

_model = whisper.load_model("turbo")

def save_temp_audio(audio_bytes):
    path = f"/tmp/audio_{uuid.uuid4().hex}.wav"
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path

def detect_language_mix(text_or_path: str) -> str:
    if text_or_path.endswith(".wav") or text_or_path.endswith(".mp3"):
        audio = whisper.load_audio(text_or_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(_model.device)
        _, probs = _model.detect_language(mel)
        lang_code = max(probs, key=probs.get)
        print(f"[LANG DEBUG] Whisper detected language code: {lang_code}")
        return "arabic" if lang_code.startswith("ar") else "english"
    else:
        lang = detect(text_or_path)
        return "arabic" if lang.startswith("ar") else "english"