import json
import logging
from datetime import datetime

import requests

import prompts as prompts
import tools as tools
from config_loader import (
    OLLAMA_URL,
)
from database import DatabaseManager

logger = logging.getLogger("knowledge_manager")


def build_first_message(therapy_json):
    therapy = json.loads(therapy_json)
    first_message = (
        f"Hi I'm your therapy management assistant!  \n"
        f"The current patient is **{therapy['patient_full_name']}**. "
        f"The activities of {therapy['patient_full_name']}'s therapy are:  \n"
    )

    days_map = {
        1: "Mon",
        2: "Tue",
        3: "Wed",
        4: "Thu",
        5: "Fri",
        6: "Sat",
        7: "Sun",
    }

    for act in therapy["activities"]:
        first_message += f"- {act['name']}   -  {act['time']}  -  {', '.join(days_map[d] for d in act.get('day_of_week', []))} \n"
    first_message += "\n"

    if len(therapy["expired_activities"]) > 0:
        first_message += "The activities that are **not valid anymore** are:  \n"
        for inv_act in therapy["expired_activities"]:
            first_message += f"- {inv_act['time']} {inv_act['name']}  -  Valid until: {inv_act['valid_until']}  \n"
        first_message += "\n"
    first_message += "I can help you add new activity, change the the current activities or remove the one that are not necessary. What do you want to do?"

    return first_message


class OllamaChat:
    def __init__(
        self,
        model="llama3",
        base_url=OLLAMA_URL,
        system_prompt=None,
        database_manager: DatabaseManager = None,
    ):
        """
        Inizializza il client Ollama

        Args:
            model: Nome del modello da utilizzare (default: llama3)
            base_url: URL base dell'API Ollama (default: http://localhost:11434)
            system_prompt: Prompt di sistema per configurare il comportamento del modello
        """
        self.model = model
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/chat"
        self.conversation_history = []

        self.database_manager = database_manager

        logger.info(f"[INIT] Model {model}")

        self.tools = tools.tools_decl
        logger.info(f"[INIT] Loaded {len(self.tools)} tools")

        if system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": system_prompt}
            )
            logger.info(
                f"[INIT] System prompt configured: {len(system_prompt)} characters"
            )
            logger.debug(f"[SYSTEM_PROMPT] {system_prompt}")
        else:
            logger.info("[INIT] No system prompt provided")

        # Initialization of useful data
        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"get_current_datetime:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}",
            }
        )
        therapy_json = tools.get_all_activities()
        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"get_all_activities:{therapy_json}",
            }
        )

        first_message = build_first_message(therapy_json)
        self.conversation_history.append(
            {"role": "assistant", "content": first_message}
        )

    def execute_tool(self, tool_name, tool_arguments):
        """
        Esegue un tool in base al nome

        Args:
            tool_name: Nome del tool da eseguire
            tool_arguments: Argomenti del tool (dict)

        Returns:
            str: Risultato dell'esecuzione del tool
        """

        if tool_name == "get_current_datetime":
            now = datetime.now()
            result = now.strftime("%Y-%m-%d %H:%M:%S %A")
            return result

        elif tool_name == "get_devices":
            result = tools.get_devices()
            return result

        elif tool_name == "clear_conversation_history":
            keep_system = tool_arguments.get("keep_system_prompt", True)
            return tools.clear_conversation_history(self, keep_system)

        # THERAPY MANAGEMENT TOOLS
        elif tool_name == "get_therapy_activities":
            return tools.get_all_activities()

        elif tool_name == "add_therapy_activity":
            return tools.add_therapy_activity(tool_arguments)

        elif tool_name == "update_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            updates = tool_arguments.get("updates", {})
            return tools.update_therapy_activity(activity_id, updates)

        elif tool_name == "remove_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            return tools.remove_therapy_activity(activity_id)

        elif tool_name == "get_medicine_data":
            medicine_name = tool_arguments.get("medicine_name")
            return tools.get_medicine_data(medicine_name)

        elif tool_name == "save_session":
            return tools.save_session(self.database_manager)

        else:
            logger.warning(f"[TOOL] Tool not found: {tool_name}")
            return f"Tool '{tool_name}' non trovato"

    def send_message(self, user_message):
        """
        Invia un messaggio al modello mantenendo il contesto della conversazione
        Gestisce automaticamente le chiamate ai tools

        Args:
            user_message: Il messaggio dell'utente da inviare

        Returns:
            str: La risposta del modello, o None in caso di errore
        """
        logger.info(f"[CHAT] USER: {user_message}")

        self.conversation_history.append({"role": "user", "content": user_message})

        # LLM can call at most 5 tools in a row
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            payload = {
                "model": self.model,
                "messages": self.conversation_history,
                "tools": self.tools,
                "stream": False,
            }

            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=120)

                response.raise_for_status()

                result = response.json()
                message = result.get("message", {})

                tool_calls = message.get("tool_calls")

                if tool_calls:
                    # Model requires tool calling
                    logger.debug(f"[TOOL] {len(tool_calls)} tool call(s) requested")
                    self.conversation_history.append(message)

                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name")
                        tool_arguments = function.get("arguments", {})

                        logger.debug(f"[TOOL] Executing: {tool_name}({tool_arguments})")

                        # Running the tool
                        tool_result = self.execute_tool(tool_name, tool_arguments)
                        result_preview = (
                            str(tool_result)[:100] + "..."
                            if len(str(tool_result)) > 100
                            else str(tool_result)
                        )
                        logger.debug(f"[TOOL] Result: {result_preview}")

                        self.conversation_history.append(
                            {"role": "tool", "content": tool_result}
                        )

                    # continue #this continue can be used if the llm needs to call multiple tools in sequence. In the case in which it needs a result first to call another  one

                else:
                    assistant_message = message.get(
                        "content", "Nessuna risposta ricevuta"
                    )
                    self.conversation_history.append(
                        {"role": "assistant", "content": assistant_message}
                    )

                    return assistant_message

            except requests.exceptions.ConnectionError:
                logger.error("[ERROR] Connection failed: Cannot connect to Ollama")
                return None
            except requests.exceptions.Timeout:
                logger.error("[ERROR] Request timeout")
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"[ERROR] Request failed: {str(e)}")
                return None
            except json.JSONDecodeError:
                logger.error("[ERROR] Invalid JSON response from server")
                return None

        logger.warning("[WARNING] Max iterations reached for tool calling")
        return "Errore: troppe chiamate ai tools"

    def get_history(self):
        return self.conversation_history
