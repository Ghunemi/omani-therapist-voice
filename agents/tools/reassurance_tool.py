import os
import random
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    openai_available = True
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    openai_available = False

class InputState(BaseModel):
    emotion: str
    intent: str
    lang_type: Literal["arabic", "english"]

# Culturally grounded fallback reassurance map
fallback_map = {
    "arabic": {
        "depression": [
            "ترى حتى الضيق يزول، والصبر مفتاح الفرج.",
            "كل إنسان يمر بلحظات ضعف، خذها فرصة تقرب فيها من ربك.",
            "إذا ضاقت بك الدنيا، فربك أرحم وأعلم بحالك."
        ],
        "anxiety": [
            "قل: حسبي الله ونعم الوكيل، وخل قلبك مطمئن.",
            "اللي مكتوب لك ما راح يفوتك، توكل على الله واهدأ.",
            "خذ نفس عميق واذكر الله، ترى القلق ما يحل الأمور."
        ],
        "family_issues": [
            "الأهل نعمة، حاول تهدي وتفتح مجال للكلام الطيب.",
            "كل بيت يمر بمشاكل، الصبر والحكمة مفتاح الحل.",
            "ادع ربك يهدي القلوب، وخل نيتك دايمًا صافية."
        ],
        "academic_stress": [
            "التعب بيروح ونتيجة تعبك بتفرحك، وراك خير كثير.",
            "نظم وقتك وخذ راحة، ولا تنسى تستخير ربك.",
            "أنت قادر، بس خفف الضغط على نفسك شوية."
        ],
        "relationship_issues": [
            "الحب الصادق يبقى، بس لازم تفهم مشاعرك بهدوء.",
            "العلاقات تمر بلحظات ضعف، لا تحكم من الزعل.",
            "جرب تكون صريح مع نفسك، وادع الله يبين لك الصح."
        ],
        "financial_stress": [
            "الرزق بيد الله، وكل ضيق له فرج.",
            "ما دامك تحاول، فربي ما رح يضيعك.",
            "خلك واقعي، وابدأ بخطوة صغيرة بتفتح لك أبواب."
        ],
        "spiritual_doubt": [
            "كلنا نمر بلحظات ضعف إيماني، بس ربك ما ينسى عباده.",
            "ارجع لصلاتك بهدوء، وتكلم مع الله كأنك تشتكي لصديق.",
            "لا تخجل من ضعفك، الإيمان يزيد ويقل، والله رحيم بعباده."
        ],
        "grief": [
            "البقاء لله، وربي يعينك على الصبر والرضا.",
            "الفقيد في رحمة الله، وذكراه تبقى في قلبك.",
            "الحزن طبيعي، بس لا تنسى تدعي وتستغفر له."
        ],
        "anger": [
            "الغضب شي طبيعي، بس حاول تهدي وتفكر قبل ما ترد.",
            "خذ لك وقت تتنفس وتستوعب، لا تخلي الغضب يسيطر.",
            "اذكر الله، ترى الذكر يطمن القلب ويهدي النفس."
        ],
        "default": [
            "ما في ضيق إلا وله فرج، وربي معك دايمًا.",
            "كلنا نمر بأوقات صعبة، بس نقدر نعديها مع بعض.",
            "اطمئن، اللي تحس فيه طبيعي، ومو عيب تطلب مساعدة."
        ]
    },
    "english": {
        "depression": [
            "Sadness doesn’t last forever — you are stronger than this moment.",
            "Even when you feel low, God’s mercy surrounds you.",
            "Take it one step at a time — light follows darkness."
        ],
        "anxiety": [
            "Breathe deeply — you are not alone in this feeling.",
            "Place your trust in God, and peace will follow.",
            "This storm will pass — give yourself grace."
        ],
        "family_issues": [
            "Every home has its struggles — approach with patience and kindness.",
            "Open-hearted conversations can heal deep wounds.",
            "Seek wisdom, stay calm — families are worth mending."
        ],
        "academic_stress": [
            "Your hard work matters — one day it will pay off.",
            "You don’t need to be perfect, just persistent.",
            "Rest is part of the journey — take breaks mindfully."
        ],
        "relationship_issues": [
            "It’s okay to feel confused — give your heart space.",
            "Love is a journey — take time to reflect and grow.",
            "Honest conversations can bring clarity and peace."
        ],
        "financial_stress": [
            "Struggles with money are heavy — but not hopeless.",
            "One step at a time — you are not defined by your income.",
            "God provides in ways we do not expect — keep hope alive."
        ],
        "spiritual_doubt": [
            "Even prophets had questions — doubt can lead to deeper faith.",
            "Reconnect gently — faith grows through honesty and prayer.",
            "God understands your struggles — don’t carry them alone."
        ],
        "grief": [
            "Grief is love that has nowhere to go — let it speak.",
            "May God comfort your heart in this time of loss.",
            "You carry their memory in your prayers and heart."
        ],
        "anger": [
            "Pause, breathe, and remind yourself who you want to be.",
            "Your anger is valid — but peace is more powerful.",
            "Step back before reacting — calm brings clarity."
        ],
        "default": [
            "You are heard, and your feelings matter.",
            "This too shall pass — trust the process.",
            "Speak what you feel — healing starts there."
        ]
    }
}

# Few-shot examples for the model
FEW_SHOTS = {
    "arabic": [
        "إذا حاسس بالحزن، تذكّر أن الله ما ينسى عباده.",
        "لو تحس بالخوف، قل: حسبي الله ونعم الوكيل.",
        "لو تمر بضغط، خذ نفس واهدأ، وخل الأمور على الله."
    ],
    "english": [
        "If you're feeling down, remember that healing takes time.",
        "When you're anxious, take a breath and trust in God's plan.",
        "For moments of confusion, pause and center yourself."
    ]
}

def generate_prompt(state: InputState) -> str:
    lang = state.lang_type
    prompt = f"User is experiencing emotion '{state.emotion}' with intent '{state.intent}'.\n"
    prompt += "Generate 1 sentence of empathetic therapeutic reassurance.\n"
    prompt += "The tone should match Omani cultural values, and use religious/spiritual references where appropriate.\n"
    prompt += f"Language: {'Arabic (Omani dialect)' if lang == 'arabic' else 'English'}\n"
    prompt += "\nHere are some examples:\n"
    for ex in FEW_SHOTS[lang]:
        prompt += f"- {ex}\n"
    prompt += "\nNow generate the final sentence:"
    return prompt

def get_static_reassurance(state: InputState) -> str:
    lang = state.lang_type
    intent = state.intent.lower()
    fallbacks = fallback_map.get(lang, {}).get(intent, []) or fallback_map.get(lang, {}).get("default", [])
    return random.choice(fallbacks)

def get_reassurance(state: InputState) -> str:
    if openai_available and os.getenv("OPENAI_API_KEY"):
        try:
            prompt = generate_prompt(state)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an empathetic Omani therapist providing faith-centered, culturally-sensitive reassurance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=60
            )
            reassurance = response.choices[0].message.content.strip()
            return reassurance
        except Exception:
            return get_static_reassurance(state)
    else:
        return get_static_reassurance(state)
