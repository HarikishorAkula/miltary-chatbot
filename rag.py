from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def create_rag_chain(api_key):

    # Load Document
    loader = TextLoader("mil.txt", encoding="utf-8")
    documents = loader.load()

    # Split Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS Vector Store
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a Military Knowledge Assistant.

Rules:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- Do NOT guess.
- If the answer is not found in the context, reply exactly:

I don't know based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG Chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain