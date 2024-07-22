import streamlit as st
from agent import financial_react_agent
from io import StringIO

AGENT_NAME = "Finvest AI"

# Streamlit UI configuration
st.set_page_config(
    page_title=AGENT_NAME,
    page_icon="💸💰",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/adiagarwalrock/finvest-ai/issues",
        "About": "Finvest AI leverages cutting-edge AI technology to provide instant and comprehensive financial insights.",
    },
)

st.title(f"{AGENT_NAME} 💸💰")
st.caption(f"🤖 Chat with {AGENT_NAME} 🔌 by Y-Finance and Llama3")

st.divider()

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"Hi! I am your friendly neighborhood financial analyst",
        }
    ]


if "react_agent" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.react_agent = financial_react_agent

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # Initialize the status container
        status = st.status("Analyzing...", expanded=True, state="running")

        response_stream = st.session_state.react_agent.stream_chat(prompt)

        # Update the status container during response generation
        # status.update(
        #     label="Still Analyzing...",
        #     expanded=True,
        #     state="running",
        # )

        st.write_stream(response_stream.response_gen)

        message = {"role": "assistant", "content": response_stream.response}

        st.session_state.messages.append(message)

        # Mark the status as complete
        status.update(label="Analysis⬇️", expanded=True, state="complete")


with st.sidebar:

    st.header("File Based Portfolio Analysis")
    # st.subheader("Choose a file📃 you want to include in analysis!")
    st.divider()
    uploaded_file = st.file_uploader(
        label="Choose a file📃 you want to include in analysis",
        accept_multiple_files=False,
        type=["csv", "xlsx", "pdf", "txt"],
        # label_visibility=False,
    )

    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)
