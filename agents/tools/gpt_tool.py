# agents/tools/gpt_tool.py

from typing import Dict, List
from agents.tools.reassurance_tool import get_reassurance, InputState
from agents.tools.emotion_intent_tool import detect_emotion_and_intent

# 🧠 Chat history
MAX_TURNS = 8
conversation_history: List[Dict[str, str]] = []

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3:instruct")


# Truncate memory
def truncate_history(history: List[Dict[str, str]], max_turns: int = MAX_TURNS) -> List[Dict[str, str]]:
    return history[-max_turns:]

# Prompt builder with language-aware instructions
def build_prompt(history: List[Dict[str, str]], lang_type: str) -> str:
    if lang_type == "arabic":
        instructions = (
            "أنت معالج نفسي عماني حنون وداعم. "
            "تقدم العلاج بلغة عربية بسيطة وبأسلوب يعكس القيم والثقافة العمانية. "
            "كن متعاطفًا، مختصرًا، وقدم نصائح عملية عند الحاجة.\n"
        )
    else:
        instructions = (
            "You are a warm, empathetic therapist from Oman. "
            "Your therapy is culturally grounded, emotionally intelligent, and suitable for Omani social values. "
            "Speak in English. Be concise, emotionally supportive, and provide practical advice if needed.\n"
        )

    full_chat = instructions
    for msg in history:
        role = "User" if msg["role"] == "user" else "Therapist"
        full_chat += f"{role}: {msg['content']}\n"
    full_chat += "Therapist:"
    return full_chat

# Main therapist responder
def agentic_therapist_response(input_dict: Dict[str, str]) -> str:
    global conversation_history

    user_text = input_dict["text"]
    analysis = detect_emotion_and_intent({"text": user_text})
    emotion = analysis["emotion"]
    intent = analysis["intent"]
    lang_type = analysis["lang_type"]

    conversation_history.append({"role": "user", "content": user_text})
    prompt = build_prompt(conversation_history, lang_type)
    print("🧠 [GPT TOOL] Final Prompt:\n", prompt)


    try:
        reply = llm.invoke(prompt).strip()
    except Exception:
        reply = (
            "عذرًا، حدث خلل في الاتصال. حاول مرة أخرى." if lang_type == "arabic"
            else "Sorry, something went wrong. Please try again."
        )

    # Add optional reassurance
    if intent in ["depression", "anxiety", "grief", "spiritual_doubt", "fear"]:
        reassurance = get_reassurance(InputState(emotion=emotion, intent=intent, lang_type=lang_type))
        reply += f"\n\n{reassurance}"

    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history = truncate_history(conversation_history)
    return reply
