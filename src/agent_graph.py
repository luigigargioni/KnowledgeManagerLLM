# agent_graph.py

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from llm_client import make_sim_client
from utils import is_exit_message


class TherapyState(TypedDict):
    messages: Annotated[list, add_messages]
    session_ended: bool


def build_therapy_graph(chat, caregiver):
    """
    Build the LangGraph that coordinates CaregiverAgent and Chat.
    """
    # The simulated user talks to its own backend (SIM_LLM), exactly as in the
    # batch runner; `chat` keeps the backend of the system under test.
    sim_client = make_sim_client()

    def caregiver_node(state: TherapyState) -> dict:

        last_message = state["messages"][-1].content

        caregiver.conversation_history.append({"role": "user", "content": last_message})
        response = sim_client.chat.completions.create(
            messages=caregiver.conversation_history,
        )
        caregiver_message = response.choices[0].message.content or ""
        caregiver.conversation_history.append({"role": "assistant", "content": caregiver_message})

        return {"messages": [HumanMessage(content=caregiver_message)]}

    def therapy_manager_node(state: TherapyState) -> dict:
        user_message = state["messages"][-1].content
        response = chat.send_message(user_message)
        return {"messages": [AIMessage(content=response)]}

    def should_continue(state: TherapyState) -> str:
        # An exit on the caregiver's opening message is a slip of the simulation,
        # not the end of the conversation: is_exit_message matches a *trailing*
        # keyword, so a first message that states the request and then appends
        # "exit" ends the run before the assistant has said anything. Measured on
        # gpt-oss-20b. `messages` holds the assistant's greeting plus the
        # caregiver's reply at this point, hence the length test. Kept in step
        # with the same guard in test.py: the two drivers have to behave alike.
        if len(state["messages"]) > 2 and is_exit_message(state["messages"][-1].content):
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
