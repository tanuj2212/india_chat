import streamlit as st
from app import qa_chain # Import the chain from your script

st.title("🇮🇳 Indian Creator Chatbot")
st.write("Ask me anything based on recent videos from your favorite creators.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a video topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = qa_chain.invoke(prompt)["result"]
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})