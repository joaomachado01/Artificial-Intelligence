# Web Interface for Chat Agent with LangChain and LLM

#######################
### Import packages ###
#######################
# Framework for web applications
import streamlit as st

# For creation and execution of chat agents
from langchain.agents import ConversationalChatAgent, AgentExecutor

# Callback for Streamlit Interface Interaction
from langchain_community.callbacks import StreamlitCallbackHandler

# OpenAI LLM Integration
from langchain_openai import ChatOpenAI

# Memory to host chat history
from langchain.memory import ConversationBufferMemory

# Streamlit History Messages
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# DuckDuckGo search tool
from langchain_community.tools import DuckDuckGoSearchRun

# Data manipulation in JSON format
import json

# Request package
import requests

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

#######################
###   Creating app  ###
#######################
# Page title
st.set_page_config(page_title = 'ChatAgent')

# Page layout cols
col1, col2 = st.columns([4, 1])

# Column for title
with col1:
    st.title("Web Interface for Chat Agent with LangChain and LLM")

# OpenAI API KEY
openai_api_key = st.sidebar.text_input("OpenAI api key", type = "password")

# Initiate message history
msgs = StreamlitChatMessageHistory()

# Chat Memory Config
memory = ConversationBufferMemory(
    chat_memory = msgs,
    return_messages = True,
    memory_key = "chat_history",
    output_key = "output"
)

# Clean history when no messages
if len(msgs.messages) == 0 or st.sidebar.button("Reset"):
    msgs.clear()
    msgs.add_ai_message("How can I help you today Sr?")
    st.session_state.steps = {}

# Choosing avatars for messages interaction
avatars = {"human": "user", "ai": "assistant"}

# Loop to show messages
# Iterate on each message
for idx, msg in enumerate(msgs.messages):

    # Creates a chat message according to the user type
    with st.chat_message(avatars[msg.type]):

        # Iterate over saved steps for each message, if any
        for step in st.session_state.steps.get(str(idx), []):

            # If exception, continue
            if step[0].tool == "_Exception":
                continue

            # Create expander for each tool used, showing the result
            with st.expander(f" **{step[0].tool}**: {step[0].tool_input}"):

                # Show tool log
                st.write(step[0].log)

                # Show tool execution result
                st.write(f"**{step[1]}**")

        # Show message content
        st.write(msg.content)

# Entry point for user messages
if prompt := st.chat_input(placeholder = "Write a question to start!"):
    st.chat_message("user").write(prompt)

    # API Verification
    if not openai_api_key:
        st.info("Please add your OpenAI API Key to continue.")
        st.stop()

    # LLM configuration
    llm_openai = ChatOpenAI(openai_api_key = openai_api_key, streaming = True)

    # Agent search tool
    search_tool = [DuckDuckGoSearchRun(name = "Search")]

    # Agent creation
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm = llm_openai, tools = search_tool)

    # Agent executor (with memory and error handling)
    executor = AgentExecutor.from_agent_and_tools(
        agent = chat_agent,
        tools = search_tool,
        memory = memory,
        return_intermediate_steps = True,
        handle_parsing_errors = True
    )

    # Assistant answer
    with st.chat_message("assistant"):

        # Callback for Streamlit
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
        response = executor(prompt, callbacks = [st_cb])
        st.write(response["output"])

        # Saving steps
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]