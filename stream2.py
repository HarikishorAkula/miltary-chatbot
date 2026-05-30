
import os
import time
from dotenv import load_dotenv
import streamlit as st

from rag import create_rag_chain

load_dotenv()

st.set_page_config(
    page_title="Military AI Assistant",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

h1 {
    text-align: center;
}

.chat-header {
    background: linear-gradient(90deg,#1e293b,#334155);
    padding: 1rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown("""
<div class="chat-header">
<h1>🪖 Military AI Assistant</h1>
<p>RAG Chatbot powered by FAISS + Groq + LangChain</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/684/684908.png",
        width=100
    )

    st.title("Military AI")

    st.markdown("---")

    st.markdown("""
### Features

✅ FAISS Vector Database

✅ Groq LLM

✅ LangChain RAG

✅ Military Knowledge Base

✅ Streamlit UI
""")

    st.markdown("---")

    st.info(
        "Ask questions only from the uploaded military document."
    )

# =====================================
# API KEY
# =====================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        st.error("GROQ API Key not found")
        st.stop()

# =====================================
# LOAD CHAIN
# =====================================

@st.cache_resource
def load_chain():
    return create_rag_chain(api_key)

chain = load_chain()

# =====================================
# STATS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Vector DB", "FAISS")

with col2:
    st.metric("LLM", "Groq")

with col3:
    st.metric("Status", "Online")

st.markdown("---")

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================
# CHAT INPUT
# =====================================

question = st.chat_input(
    "Ask about Army, Navy, Air Force, Weapons..."
)

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

        placeholder = st.empty()

        full_response = ""

        try:

            answer = chain.invoke(question)

            for word in answer.split():

                full_response += word + " "

                placeholder.markdown(full_response)

                time.sleep(0.02)

        except Exception as e:

            answer = f"Error: {str(e)}"

            placeholder.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
    "<div class='footer'>Built with Streamlit • LangChain • Groq • FAISS</div>",
    unsafe_allow_html=True
)
