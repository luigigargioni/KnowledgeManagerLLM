from agents.agent import Agent

_PROMPT = """
You are a caregiver user tasked to manage patients' therapies and activities.
Your general job consist in interacting with a chatbot to request the creation, update and deletion of activities only in natural language.
You are a USER of the system, so you cannot do actions directly but you always need to delegate them to the chatbot that will serve as your assistant.
Do act and speak like a proper user not an assistant. YOU ARE NOT THE ASSISTANT HERE.

In order to help you with your task, you are given a scenario that tells you which are you objective and actions you are supposed to perform.
Please do interact with the chatbot with the sole objective of completing the script. If there are multiple objectives, please ask them one by one and move to the next one only after the previous was finished.
Once you think you have completed your task send a message only containing the "exit" word to end the conversation.

{script}
"""


class CaregiverAgent(Agent):
    def __init__(self, agent_name="caregiver_agent", zero_shot=False, script=""):
        agent_prompt = _PROMPT.format(script=script)
        super().__init__(
            agent_name=agent_name,
            agent_prompt=agent_prompt,
            agent_tools={},
            zero_shot=zero_shot,
        )

    def inject_context(self):
        return super().inject_context()

    # therapy_json = tools.get_all_activities()
    # self.conversation_history.append(
    #    {
    #        "role": "system",
    #        "content": f"Current patient context:{therapy_json}",
    #    }
    # )
