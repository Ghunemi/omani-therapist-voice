from typing import Literal
from pydantic import BaseModel

class PlannerInput(BaseModel):
    text: str
    emotion: str
    intent: str
    lang_type: Literal["arabic", "english"]

# Keywords for fallback
crisis_keywords = {
    "suicidal", "suicide", "kill myself", "harm myself", "self harm", "end my life", "want to die",
    "can't go on", "no reason to live", "wish I was dead", "die", "die die die", "jump", "cut myself",
    "انتحار", "قتل", "أؤذي نفسي", "أذى نفسي", "أبي أموت", "أموت", "ما أتحمل", "ودي أختفي",
    "أقطع وريدي", "أرمي نفسي", "مليت", "زهقت من حياتي", "أرتاح من الدنيا", "حاسس بضيق", "نفسي أموت"
}


cbt_fallback_terms = {
    "depression", "anxiety", "panic", "fear", "worried", "scared", "overwhelmed", "mental health",
    "sadness", "stress", "pressure", "hopeless", "can't sleep", "crying", "breakdown",

    "توتر", "قلق", "مكتئب", "خوف", "ضغط", "كآبة", "مش قادر", "دموعي", "منهار", "نفسيتي تعبانة", "تعبان نفسياً",
    "مش عارف أنام", "حاسس بكآبة", "قلقان", "خايف", "مرعوب"
}


spiritual_terms = {
    "الله", "ربنا", "صلاة", "ذنب", "إيمان", "تقصير", "دعاء", "ذكر", "استغفار", "جنة", "نار", "مغفرة", "توبة",
    "حسيت بتقصير", "حاسس بذنب", "محتاج أتقرب", "علاقتي بربنا", "ابتعدت عن الدين", "مش بصلي",

    "god", "allah", "faith", "pray", "prayer", "guilt", "repent", "forgive", "spiritual", "lost faith",
    "feel guilty", "god abandoned me", "i stopped praying", "don't feel connected spiritually"
}


def plan(state: dict) -> str:
    data = PlannerInput(**state)
    lower_text = data.text.lower()

    # High risk: Crisis
    if any(term in lower_text for term in crisis_keywords):
        return "crisis_tool"

    # Primary CBT routing
    if data.intent in {"anxiety", "depression", "fear", "academic_stress", "anger"}:
        return "cbt_agent"

    # Primary spiritual routing
    if data.intent in {"spiritual_doubt", "grief", "guilt"}:
        return "spiritual_support_agent"

    # Fallback CBT routing if intent unclear but text contains CBT symptoms
    if data.intent == "general_support" and any(term in lower_text for term in cbt_fallback_terms):
        return "cbt_agent"

    # Fallback spiritual routing if text contains religious cues
    if data.intent == "general_support" and any(term in lower_text for term in spiritual_terms):
        return "spiritual_support_agent"

    # Default GPT therapist
    return "gpt_tool"
