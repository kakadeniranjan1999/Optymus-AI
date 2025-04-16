import streamlit as st
from chatbot import chat

st.set_page_config(
    page_title="Optymus AI",
    initial_sidebar_state="expanded"
)

st.title("Optymus AI")

# sets up sidebar nav widgets
with st.sidebar:
    st.markdown("# Optymus Options")

    # model = st.selectbox('What model would you like to use?', ("mistral", "deepseek-r1:8b"))
    model = st.selectbox('What model would you like to use?', ("mistral"))

# checks for existing messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("What would you like to ask?"):
    # Display user prompt in chat message widget
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # adds user's prompt to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner('Generating response...', show_time=True):
        # retrieves response from model
        llm_stream = chat(user_prompt, model=model)

        # streams the response back to the screen
        stream_output = st.write_stream(llm_stream)

        # appends response to the message list
        st.session_state.messages.append({"role": "assistant", "content": stream_output})