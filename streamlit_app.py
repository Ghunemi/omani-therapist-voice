# streamlit_app.py

import streamlit as st
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import uuid
import requests
from dotenv import load_dotenv

from agents.graph_agent import build_graph
from agents.tools.whisper_tool import transcribe_audio
from utils.helpers import detect_language_mix

# Load environment variables
load_dotenv()
AZURE_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_REGION = os.getenv("AZURE_TTS_REGION")

# Initialize LangGraph
graph = build_graph()

st.set_page_config(page_title="OMANI Therapist", page_icon="🧠", layout="centered")
st.title("🧠 OMANI Therapist (Voice-Only Demo)")
st.markdown("Speak freely, and receive culturally aware therapy powered by AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Azure Text-to-Speech
def synthesize_azure(text, lang_type):
    if not AZURE_KEY or not AZURE_REGION:
        st.error("🔴 Azure credentials missing")
        return None

    voice = "ar-OM-RashidNeural" if lang_type == "arabic" else "en-US-GuyNeural"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
    }
    url = f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    ssml = f"""
    <speak version='1.0' xml:lang='{"ar-OM" if lang_type == "arabic" else "en-US"}'>
        <voice xml:lang='{"ar-OM" if lang_type == "arabic" else "en-US"}' xml:gender='Male' name='{voice}'>
            {text}
        </voice>
    </speak>
    """

    try:
        response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
        if response.status_code == 200:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp.write(response.content)
            temp.close()
            return temp.name
        else:
            st.error("🔴 Azure TTS failed. Check credentials or quota")
            return None
    except Exception as e:
        st.error(f"🔴 Azure TTS exception: {e}")
        return None

# Voice recording and processing
if st.button("🎙️ Record Voice"):
    duration = 7  # seconds
    st.info("Recording... Please speak now")
    recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()

    temp_path = f"/tmp/audio_{uuid.uuid4().hex}.wav"
    sf.write(temp_path, recording, 16000)
    st.audio(temp_path, format='audio/wav')

    # Transcribe
    transcript, whisper_lang = transcribe_audio(open(temp_path, "rb").read())
    lang_type = "arabic" if whisper_lang.startswith("ar") else "english"

    st.success(f"You said: {transcript}")

    # LangGraph state and response
    # Step 1: Append user input to session history first
    st.session_state.messages.append({"role": "user", "content": transcript})

    # Step 2: Send updated history to graph
    state = {
        "messages": st.session_state.messages,
        "emotion": "",
        "intent": "",
        "lang_type": ""
    }

    output = graph.invoke(state)

    # Step 3: Append assistant reply
    assistant_reply = output["messages"][-1]["content"]
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    st.markdown(f"🧑‍⚕️: {assistant_reply}")

    # Text-to-Speech Output
    audio_path = synthesize_azure(assistant_reply, lang_type)
    if audio_path:
        st.audio(audio_path, format="audio/mp3")
