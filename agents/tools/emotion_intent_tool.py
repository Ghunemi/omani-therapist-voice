import sys, os
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.helpers import detect_language_mix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Arabic emotion model
arabic_model = "alpcansoydas/bert-base-arabic-emotion-analysis-v2"
arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model)
arabic_emotion_model = AutoModelForSequenceClassification.from_pretrained(arabic_model)
arabic_classifier = pipeline("text-classification", model=arabic_emotion_model, tokenizer=arabic_tokenizer, top_k=1)

# English emotion model
english_model = "j-hartmann/emotion-english-distilroberta-base"
english_tokenizer = AutoTokenizer.from_pretrained(english_model)
english_emotion_model = AutoModelForSequenceClassification.from_pretrained(english_model)
english_classifier = pipeline("text-classification", model=english_emotion_model, tokenizer=english_tokenizer, top_k=1)

# Arabic intent keywords
intent_keywords_ar = {
    "family_issues": ["عائلة", "زوج", "زوجة", "طلاق", "مشاكل أسرية", "أم", "أب", "أولاد"],
    "anxiety": ["قلق", "خوف", "توتر", "هلع", "رهبة"],
    "work_stress": ["شغل", "وظيفة", "مدير", "دوام", "ضغط العمل"],
    "academic_stress": ["دراسة", "جامعة", "مدرسة", "امتحان", "مشروع تخرج"],
    "social_isolation": ["وحدة", "انعزال", "ما عندي أصحاب", "بجلس لحالي", "كئيب"],
    "spiritual_doubt": ["إيمان", "دين", "الله", "صلاة", "ربنا", "الخالق", "تقصير", "ذنب", "غفران"],
    "crisis": ["انتحار", "أؤذي نفسي", "أموت", "أذى", "مش طايق أعيش"],
    "depression": ["اكتئاب", "مكتئب", "حزين", "تعيس", "ما في أمل", "محبط", "فقدت الأمل"]
}

# English intent keywords
intent_keywords_en = {
    "family_issues": ["wife", "husband", "divorce", "mother", "father", "family", "kids"],
    "anxiety": ["anxiety", "worried", "panic", "afraid", "scared", "nervous", "uneasy"],
    "work_stress": ["job", "work", "boss", "deadline", "overworked"],
    "academic_stress": ["exam", "school", "university", "grades", "homework", "project"],
    "social_isolation": ["lonely", "no friends", "isolated", "alone"],
    "spiritual_doubt": ["god", "faith", "religion", "sin", "forgiveness", "prayer"],
    "crisis": ["suicide", "kill myself", "hurt myself", "want to die", "ending my life"],
    "depression": [
        "depression", "depressed", "no hope", "hopeless", "sad", "miserable",
        "i feel down", "feeling down", "i am down", "feeling low", "down", "unmotivated", "worthless"
    ]
}

def detect_intent(text: str, lang_type: str) -> str:
    keywords = intent_keywords_ar if lang_type == "arabic" else intent_keywords_en
    text_lower = text.lower()

    for intent, words in keywords.items():
        for word in words:
            if word.lower() in text_lower:
                return intent

    # Extra heuristic fallback
    if lang_type == "english" and "depress" in text_lower:
        return "depression"
    if lang_type == "arabic" and "اكتئاب" in text_lower:
        return "depression"

    return "general_support"

def detect_emotion_and_intent(state: Dict[str, str]) -> Dict[str, str]:
    text = state["text"]
    lang_type = detect_language_mix(text)

    # Emotion classification
    if lang_type == "arabic":
        emotion = arabic_classifier(text)[0][0]["label"]
    else:
        emotion = english_classifier(text)[0][0]["label"]

    # Intent classification
    intent = detect_intent(text, lang_type)

    # Debug log
    print(f"[DEBUG] Text: {text}")
    print(f"[DEBUG] Detected emotion: {emotion}")
    print(f"[DEBUG] Detected intent: {intent}")
    print(f"[DEBUG] Lang type: {lang_type}")

    return {
        "emotion": emotion,
        "intent": intent,
        "lang_type": lang_type,
        "text": text
    }
