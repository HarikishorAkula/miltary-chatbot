from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(api_key):

    loader = TextLoader("mil.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Military Knowledge Assistant.

Rules:
1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. If the answer is not present in the context, reply exactly:
   I don't know based on the provided context.
5. Keep answers concise.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

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
