from typing import Literal, List, Dict
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3:instruct")

MAX_TURNS = 8

def truncate_history(history: List[Dict[str, str]], max_turns: int = MAX_TURNS) -> List[Dict[str, str]]:
    return history[-max_turns:]

def build_spiritual_prompt(messages: List[Dict[str, str]], lang_type: str, emotion: str, intent: str) -> str:
    history = truncate_history(messages)

    if lang_type == "arabic":
        prompt = (
            "أنت مستشار روحي عماني تقدم دعمًا إيمانيًا مريحًا. "
            "استخدم مفاهيم دينية ولغة حنونة تعكس الثقافة العمانية.\n"
        )
        for msg in history:
            role = "المستخدم" if msg["role"] == "user" else "المستشار الروحي"
            prompt += f"{role}: {msg['content']}\n"
        prompt += (
            f"\nيشعر المستخدم بـ '{emotion}' ولديه نية '{intent}'. "
            f"قدم له دعمًا روحيًا مريحًا باللغة العربية فقط.\n"
            "المستشار الروحي:"
        )
    else:
        prompt = (
            "You are a compassionate Omani faith-based counselor offering Islamic spiritual comfort. "
            "Use gentle religious references and speak with emotional warmth.\n"
        )
        for msg in history:
            role = "User" if msg["role"] == "user" else "Spiritual Counselor"
            prompt += f"{role}: {msg['content']}\n"
        prompt += (
            f"\nThe user feels '{emotion}' and has the intent '{intent}'. "
            f"Provide a spiritually uplifting and calming response in English only.\n"
            "Spiritual Counselor:"
        )

    return prompt


def run_spiritual_support(state: Dict[str, any]) -> str:
    try:
        prompt = build_spiritual_prompt(
            state["messages"],
            state["lang_type"],
            state["emotion"],
            state["intent"]
        )
        print("[SPIRITUAL TOOL] Final Prompt:\n", prompt)

        return llm.invoke(prompt).strip()
    except Exception:
        return (
            "حدث خطأ أثناء تقديم الدعم الروحي. حاول مرة أخرى لاحقًا."
            if state["lang_type"] == "arabic"
            else "There was an error providing spiritual support. Please try again later."
        )