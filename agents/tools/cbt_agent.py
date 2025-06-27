# agents/tools/cbt_agent.py

from typing import Literal, List, Dict

# 🌍 Local LLM
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3:instruct")


MAX_TURNS = 8

def truncate_history(history: List[Dict[str, str]], max_turns: int = MAX_TURNS) -> List[Dict[str, str]]:
    return history[-max_turns:]

def build_cbt_prompt(messages: List[Dict[str, str]], lang_type: str, emotion: str, intent: str) -> str:
    history = truncate_history(messages)

    if lang_type == "arabic":
        prompt = (
            "أنت معالج عماني مختص في العلاج المعرفي السلوكي (CBT). "
            "قدّم دعمًا نفسيًا بأسلوب عماني وبلغة بسيطة.\n"
        )
    else:
        prompt = (
            "You are an Omani CBT therapist trained in culturally sensitive Cognitive Behavioral Therapy. "
            "Offer supportive and concise CBT-based guidance in English.\n"
        )

    for msg in history:
        role = "User" if msg["role"] == "user" else "Therapist"
        prompt += f"{role}: {msg['content']}\n"

    prompt += (
        f"\nNow the user is feeling '{emotion}' and their issue is '{intent}'. "
        f"Please respond using CBT techniques such as thought reframing, grounding, or small steps.\n"
        "Therapist:"
    )
    return prompt

def run_cbt_support(state: Dict[str, any]) -> str:
    try:
        prompt = build_cbt_prompt(
            state["messages"],
            state["lang_type"],
            state["emotion"],
            state["intent"]
        )
        print("🧠 [CBT TOOL] Final Prompt:\n", prompt)
        return llm.invoke(prompt).strip()
    except Exception:
        return (
            "صار خلل بسيط، حاول مرة ثانية وإن شاء الله يكون كل خير."
            if state["lang_type"] == "arabic"
            else "Something went wrong. Please try again later."
        )