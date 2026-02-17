import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import time

import requests

import prompts as prompts
import tools as tools
from config_loader import (
    FILE_LOG_LEVEL,
    MODEL,
    OLLAMA_URL,
    TERMINAL_LOG_LEVEL,
)
from utils import get_system_info


def setup_logger():
    """Configura il logger per scrivere su file di sessione nella cartella logs e su terminale"""

    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"chat_session_{session_timestamp}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_handler.setFormatter(file_formatter)

    # Terminal Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(TERMINAL_LOG_LEVEL)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"[SESSION] New chat session started ID:{session_timestamp}")

    return logger


logger = setup_logger()


class OllamaChat:
    def __init__(self, model="llama3", base_url=OLLAMA_URL, system_prompt=None):
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

        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"get_all_activities:{tools.get_all_activities()}",
            }
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
                    # In case the LLM keeps calling tools when its not needed a workaround is removing past calls in the history
                    # cleaned_history = []
                    # for msg in self.conversation_history:
                    #    if msg.get("role") in ["user", "system"]:
                    #        cleaned_history.append(msg)
                    #    elif msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    #        cleaned_history.append(msg)
                    # self.conversation_history = cleaned_history

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


def main():
    cpu_info, ram_info, gpu_info = get_system_info()

    logger.info(
        f"[SYS] CPU:{cpu_info['model']} Cores:{cpu_info['cores']}/{cpu_info['threads']} RAM:{ram_info:.0f} GB"
    )

    if len(gpu_info) > 0:
        for info in gpu_info:
            logger.info(f"[SYS] GPU {info['gpu']}: {info['name']} ({info['memory']})")

    # Chat data
    model_name = MODEL
    system_prompt = prompts.system

    # Chat initialization
    chat = OllamaChat(model=model_name, system_prompt=system_prompt)

    print("=" * 60)
    print("  OLLAMA CHAT - Local LLM Interface")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("Commands: 'exit' or 'quit' to end session")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            # Manual exit
            if user_input.lower() in ["exit", "quit", "esci"]:
                logger.info("[SESSION] Chat session ended by user")
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            start = time()
            response = chat.send_message(user_input)
            logger.debug(f"[TIMING] Total elapsed time: {time() - start:.2f}s")
            logger.info(f"[CHAT] ASSISTANT: {response}")

            if response:
                print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            logger.info("[SESSION] Chat interrupted by user (Ctrl+C)")
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logger.exception(f"[ERROR] Unexpected error: {str(e)}")
            print(f"\nUnexpected error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
