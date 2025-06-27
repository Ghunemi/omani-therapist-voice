import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import whisper
from utils.helpers import save_temp_audio, detect_language_mix

# Load Whisper model
model = whisper.load_model("turbo")

def transcribe_audio(audio_bytes: bytes) -> tuple[str, str]:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = model.transcribe(tmp.name, language=None)
        transcript = result["text"]
        detected_lang = result["language"]
        return transcript, detected_lang
