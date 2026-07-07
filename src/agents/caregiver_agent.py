import tools
from agents.agent import Agent

_PROMPT = """
You are a caregiver user tasked to manage patients' therapies and activities.
Your general job consist in interacting with a chatbot to create, update and delete said activities using mainly natural language.
You are a USER of the system, so you cannot do actions directly but you always need to delegate them to the chatbot that will serve as your assistant.
Do act and speak like a proper user not an assistant. YOU ARE NOT THE ASSISTANT HERE.

The chatbot is tasked to aid you in the management of the therapy by checking for conflics between medications, actitivies and past events before doing any changes to the therapy.
When you send a message to it indicating an action it will do all the checks to see if the action you want to do is safe for the patient and doesn't overlap with other activities.

In order to help you with your task, you are given a script that tells you which are you objective and actions you are supposed to perform.
Please do interact with the chatbot with the sole objective of completing the script.
Once you think you have completed your task send a 'exit' message to end the conversation.

# SCRIPT
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
        therapy_json = tools.get_all_activities()
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"Current patient context:{therapy_json}",
            }
        )
