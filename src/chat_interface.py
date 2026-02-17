from time import time

import streamlit as st

import prompts
from chat import OllamaChat
from config_loader import MODEL
from main import setup_logger
from utils import get_system_info

# Chat data
model_name = MODEL
system_prompt = prompts.system

if "logger" not in st.session_state:
    st.session_state.logger = setup_logger()
logger = st.session_state.logger

if "chat" not in st.session_state:
    chat = OllamaChat(model=model_name, system_prompt=system_prompt)
    st.session_state.chat = chat

if "conversation" not in st.session_state:
    st.session_state.conversation = []


# region streamlit
st.set_page_config(page_title="KnowledgeManagerLLM", page_icon="🤖", layout="centered")

# Titolo dell'applicazione
st.title("KnowledgeManagerLLM")


# Visualizzazione dei messaggi
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Input per nuovi messaggi (non funzionale per ora)
user_message = st.chat_input("Scrivi un messaggio...")
if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)
    st.session_state.conversation.append({"role": "user", "message": user_message})
    with st.chat_message("assistant"):
        start = time()

        with st.spinner("Thinking..."):
            response_gen = st.session_state.chat.send_message(user_message)

        st.session_state.conversation.append(
            {"role": "assistant", "message": response_gen}
        )
        with st.container():
            st.markdown(response_gen)
            st.badge(f"🕛{time() - start:.2f}s")

# Sidebar con informazioni
with st.sidebar:
    with st.expander("Info sistema"):
        cpu_info, ram_info, gpu_info = get_system_info()

        st.markdown(
            f" ⚙️ **CPU** : {cpu_info['model']} {cpu_info['cores']}/{cpu_info['threads']}"
        )

        st.markdown(f"🧠 **RAM** : {ram_info:.0f} GB")

        if len(gpu_info) > 0:
            for info in gpu_info:
                st.markdown(
                    f"🎮 **GPU {info['gpu']}** : {info['name']} ({info['memory']})"
                )


# endregion
