from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List
from agents.planner import plan, PlannerInput
from agents.tools.emotion_intent_tool import detect_emotion_and_intent
from agents.tools.reassurance_tool import get_reassurance, InputState
from agents.tools.gpt_tool import agentic_therapist_response
from agents.tools.cbt_agent import run_cbt_support
from agents.tools.spiritual_support_agent import run_spiritual_support
from agents.tools.crisis_tool import run_crisis_intervention
from agents.tools.cbt_agent import run_cbt_support
from agents.tools.spiritual_support_agent import run_spiritual_support
from dotenv import load_dotenv
load_dotenv()


class ChatState(TypedDict):
    messages: List[dict]
    emotion: str
    intent: str
    lang_type: str

#  Step 1: Classify emotion & intent
def classify(state: ChatState) -> ChatState:
    user_msg = next(m for m in reversed(state["messages"]) if m["role"] == "user")["content"]

    result = detect_emotion_and_intent({"text": user_msg})
    return {
        **state,
        "emotion": result["emotion"],
        "intent": result["intent"],
        "lang_type": result["lang_type"]
    }

#  Step 2: Decide next step using planner
def router(state: ChatState) -> str:
    latest = next(m for m in reversed(state["messages"]) if m["role"] == "user")["content"]

    lang_type = state.get("lang_type", "english")
    plan_input = PlannerInput(
        text=latest,
        emotion=state["emotion"],
        intent=state["intent"],
        lang_type=lang_type
        
    )
    router = plan(plan_input.dict())
    print('ROUTING TO:', router)
    return router

#  Step 3: Route to GPT tool
def gpt_node(state: ChatState) -> ChatState:

    user_input = next(m for m in reversed(state["messages"]) if m["role"] == "user")["content"]



    reply = agentic_therapist_response({"text": user_input})

    state["messages"].append({"role": "assistant", "content": reply})
    return state

#  Step 4: Route to CBT tool
def cbt_node(state: ChatState) -> ChatState:
    reply = run_cbt_support(state)
    state["messages"].append({"role": "assistant", "content": reply})
    return state

#  Step 5: Route to spiritual tool
def spiritual_node(state: ChatState) -> ChatState:
    reply = run_spiritual_support(state)
    state["messages"].append({"role": "assistant", "content": reply})
    return state

# Step 6: Crisis tool node
def crisis_node(state: ChatState) -> ChatState:
    user_input = next(m for m in reversed(state["messages"]) if m["role"] == "user")["content"]
    lang_type = state["lang_type"]
    reply = run_crisis_intervention(user_input, lang_type)
    state["messages"].append({"role": "assistant", "content": reply})
    return state


#  Build the LangGraph
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("classify", classify)
    graph.add_node("gpt_tool", gpt_node)
    graph.add_node("cbt_agent", cbt_node)
    graph.add_node("spiritual_support_agent", spiritual_node)
    graph.add_node("crisis_tool", crisis_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", router, {
        "gpt_tool": "gpt_tool",
        "cbt_agent": "cbt_agent",
        "spiritual_support_agent": "spiritual_support_agent",
        "crisis_tool": "crisis_tool"
    })

    graph.add_edge("gpt_tool", END)
    graph.add_edge("cbt_agent", END)
    graph.add_edge("spiritual_support_agent", END)
    graph.add_edge("crisis_tool", END)

    return graph.compile()

#  Manual test
if __name__ == "__main__":
    load_dotenv()

    app = build_graph()
    input_text = input("👤 Say something: ")
    state = {
        "messages": [{"role": "user", "content": input_text}],
        "emotion": "",
        "intent": "",
        "lang_type": ""
    }
    final_state = app.invoke(state)
    print(":", final_state["messages"][-1]["content"])