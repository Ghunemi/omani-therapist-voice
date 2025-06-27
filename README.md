# 🇴🇲 Omani Therapist Voice

**Omani Therapist Voice** is a culturally sensitive, voice-only mental health assistant that understands Arabic (including Omani dialect) and English. It uses AI to classify emotion and intent, then routes the user to the appropriate virtual therapist: GPT-based, CBT-focused, spiritual, or crisis intervention.

This project is designed as a demo and proof of concept for an agentic voice-based therapeutic system.

---

## 🚀 Features

- 🎙️ **Voice-only input and output** via real-time speech recognition (Whisper) and text-to-speech (Azure TTS)
- 🧠 **Emotion & Intent Detection** using a lightweight LLM
- 🧭 **LangGraph Agentic Planner** routes conversations to:
  - 🤖 General GPT Therapist
  - 🧘‍♀️ CBT Therapist
  - ☪️ Spiritual Support
  - 🚨 Crisis Intervention
- 💬 **Arabic-English language detection and handling**
- 📜 **Conversation history memory** (up to 8 turns)
- 🛡️ Safety & emergency awareness
- 🌐 Fully local or hybrid deployment with support for Ollama or OpenAI

---

## 🧱 Architecture

- **Streamlit**: Web interface with mic recording and real-time playback
- **LangGraph**: Planning-based agent routing engine
- **Whisper**: Converts voice to text
- **Azure TTS**: Converts response back to voice
- **OpenAI or Ollama**: Language model backend
- **Custom tools**: Crisis detector, spiritual agent, CBT support module

---

## 🧪 Test Scenarios

| User Input (Voice/Text) | Expected Route           | Agent Response Example |
|-------------------------|--------------------------|-------------------------|
| "أريد أن أموت"          | 🚨 `crisis_tool`         | تنبيه: إذا كنت تفكر...  |
| "أنا متوتر جدًا"        | 🧘 `cbt_agent`           | أفهم مشاعرك...          |
| "أشعر أني مقصر مع الله" | ☪️ `spiritual_support`   | كلنا نمر بلحظات ضعف...   |
| "I feel lonely lately"  | 🤖 `gpt_tool`            | I'm here to support you.|

---

## 🛠️ How to Run

### 1. Clone the repo
git clone https://github.com/Ghunemi/omani-therapist-voice.git
cd omani-therapist-voice

2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

4. Add .env file
Create a .env file with:

OPENAI_API_KEY=your_openai_key
AZURE_TTS_KEY=your_azure_key
AZURE_TTS_REGION=your_azure_region
⚠️ Do NOT commit .env to GitHub.

4. Run the app
streamlit run streamlit_app.py


## 🧠  Future Improvements
🤝 Add CrewAI/Autogen support for memory and multi-agent collaboration
🗂️ Store anonymized session logs
🧭 Smart planner with learning from interaction history
🧪 Add more test coverage & unit tests

##🤝 Authors
Abdelrahman Ghunemi – GitHub

##🛡️ Disclaimer
This is a proof of concept for educational purposes. It is not a replacement for real therapy or crisis support. If you or someone you know is in danger, contact professional help or emergency services immediately.
