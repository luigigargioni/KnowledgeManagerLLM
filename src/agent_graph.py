# agent_graph.py

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class TherapyState(TypedDict):
    messages: Annotated[list, add_messages]
    session_ended: bool


def build_therapy_graph(chat, caregiver):
    """
    Build the LangGraph that coordinates CaregiverAgent and Chat.
    """

    def caregiver_node(state: TherapyState) -> dict:

        last_message = state["messages"][-1].content

        caregiver.conversation_history.append({"role": "user", "content": last_message})
        response = chat.client.chat.completions.create(
            model=chat.model,
            messages=caregiver.conversation_history,
        )
        caregiver_message = response.choices[0].message.content or ""
        caregiver.conversation_history.append(
            {"role": "assistant", "content": caregiver_message}
        )

        return {"messages": [HumanMessage(content=caregiver_message)]}

    def therapy_manager_node(state: TherapyState) -> dict:
        user_message = state["messages"][-1].content
        response = chat.send_message(user_message)
        return {"messages": [AIMessage(content=response)]}

    def should_continue(state: TherapyState) -> str:
        last = state["messages"][-1].content.strip().lower()
        if last in ["exit", "quit", "esci"]:
            return "end"
        return "therapy_manager"

    builder = StateGraph(TherapyState)

    builder.add_node("caregiver", caregiver_node)
    builder.add_node("therapy_manager", therapy_manager_node)

    builder.set_entry_point("caregiver")

    builder.add_conditional_edges(
        "caregiver",
        should_continue,
        {"end": END, "therapy_manager": "therapy_manager"},
    )
    builder.add_edge("therapy_manager", "caregiver")

    return builder.compile()
