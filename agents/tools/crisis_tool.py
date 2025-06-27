import re
from typing import Optional
from pydantic import BaseModel
from agents.tools.emotion_intent_tool import detect_emotion_and_intent

# crisis phrases for both Arabic (Omani) and English
CRISIS_KEYWORDS = {
    "arabic": [
        "أفكر أنتحر", "أبي أموت", "ما أشوف فايدة من الحياة", "ودي أختفي", "أذى نفسي",
        "أقطع وريدي", "أرمي نفسي", "ما أتحمل", "انتحار", "أنهى حياتي", "موت",
        "مليت من كل شيء", "كاره الحياة", "حاسس بضيق ما له حل", "أريد أرتاح للأبد"
    ],
    "english": [
        "i want to die", "i'm suicidal", "i'm done with life", "kill myself",
        "end my life", "cut myself", "self harm", "jump off", "no reason to live",
        "can't go on", "wish i was dead", "tired of living", "i hate my life"
    ]
}

class CrisisOutput(BaseModel):
    is_crisis: bool
    detected_phrase: Optional[str] = None
    lang_type: str
    text: str

def detect_crisis(text: str, lang_type: str) -> CrisisOutput:
    lang_type = lang_type.lower()
    patterns = CRISIS_KEYWORDS.get(lang_type, [])
    for phrase in patterns:
        if re.search(re.escape(phrase), text.lower()):
            return CrisisOutput(
                is_crisis=True,
                detected_phrase=phrase,
                lang_type=lang_type,
                text=text
            )
    return CrisisOutput(is_crisis=False, lang_type=lang_type, text=text)

def get_crisis_message(lang_type: str) -> str:
    if lang_type.lower() == "arabic":
        return "🚨 إذا كنت تفكر في إيذاء نفسك، تواصل مع الطوارئ أو شخص قريب منك. حياتك غالية والله معك."
    else:
        return "🚨 If you're thinking of hurting yourself, please contact emergency services or someone you trust. Your life matters and you're not alone."

def run_crisis_intervention(text: str, lang_type: str) -> str:
    analysis = detect_emotion_and_intent({"text": text})
    crisis_check = detect_crisis(text, lang_type)
    if crisis_check.is_crisis:
        return get_crisis_message(crisis_check.lang_type)
    else:
        return "Your message doesn't appear to be a crisis, but I'm still here to support you."
