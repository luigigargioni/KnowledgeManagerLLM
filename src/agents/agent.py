import logging

logger = logging.getLogger("knowledge_manager")


class Agent:
    def __init__(
        self,
        agent_name: str = "",
        agent_prompt: str = None,
        agent_tools: list = None,
        zero_shot: bool = False,
    ):
        self.name = agent_name
        self.prompt = agent_prompt
        self.tools = agent_tools or []
        self.zero_shot = zero_shot
        self.conversation_history = []

        if agent_prompt:
            self.conversation_history.append(
                {"role": "system", "content": agent_prompt}
            )

        self.inject_context()
        logger.info(
            f"[{self.name.upper()}][INIT] Agent initialized: Agent prompt {len(self.prompt)} characters. Agent tools {len(self.tools)}"
        )

    def reset_agent(self):
        self.conversation_history = (
            [self.conversation_history[0]] if self.prompt else []
        )

    def inject_context(self):
        """Function used to inject some initial context in the agent"""
        pass

    def execute_tool(self, tool_name: str, tool_arguments: dict) -> str:
        """
        Runs a specific tool of the agent.
        The function can be overridden by subclasses to implement specific behaviors.
        """
        logger.warning(f"[{self.name}] Tool non trovato: {tool_name}")
        return f"Tool '{tool_name}' not found in agent '{self.name}'"

    def as_tool_declaration(self, description: str, parameters: dict = None) -> dict:
        """
        Convert the agent in a tool function called by the supervisor.
        """
        return {
            "type": "function",
            "function": {
                "name": f"delegate_to_{self.name}",
                "description": description,
                "parameters": parameters
                or {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send to the agent",
                        }
                    },
                    "required": ["message"],
                },
            },
        }
