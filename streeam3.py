import os
import time
import streamlit as st

from rag import create_rag_chain

# =====================================
# PAGE CONFIG
# =====================================

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

.chat-header {
    background: linear-gradient(90deg,#1e293b,#334155);
    padding: 1rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
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
<p>Powered by FAISS + Groq + LangChain</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# API KEY
# =====================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("GROQ_API_KEY not found.")
        st.stop()

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🪖 Military AI")

    st.markdown("---")

    st.markdown("""
### Features

✅ FAISS Vector Search

✅ Groq LLM

✅ LangChain RAG

✅ Military Knowledge Base

✅ Streamlit Interface
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "Ask questions only from the military document."
    )

# =====================================
# LOAD RAG CHAIN
# =====================================

@st.cache_resource
def load_chain():
    return create_rag_chain(api_key)

chain = load_chain()

# =====================================
# STATUS CARDS
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

        try:

            answer = chain.invoke(question)

            streamed_text = ""

            for word in answer.split():

                streamed_text += word + " "

                placeholder.markdown(
                    streamed_text + "▌"
                )

                time.sleep(0.01)

            placeholder.markdown(streamed_text)

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
    """
    <div class='footer'>
        Built with Streamlit • LangChain • Groq • FAISS
    </div>
    """,
    unsafe_allow_html=True
)
