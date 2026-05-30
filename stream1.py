import os
from dotenv import load_dotenv
import streamlit as st

from rag import create_rag_chain

load_dotenv()

st.set_page_config(
    page_title="Military RAG Chatbot",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Military RAG Chatbot")
st.write("Ask questions from your military knowledge base")

# Local .env OR Streamlit Secrets
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("GROQ_API_KEY not found.")
        st.stop()


@st.cache_resource
def load_chain():
    return create_rag_chain(api_key)


chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a military question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching military documents..."):

            try:
                answer = chain.invoke(question)

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )