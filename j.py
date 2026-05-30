# 1. Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ Use the new Ollama integration
from langchain_ollama import OllamaLLM

# 2. Load documents
loader = TextLoader("mil.txt")
documents = loader.load()

print("Documents loaded:", len(documents))
if len(documents) == 0:
    raise ValueError("No documents found. Please check 'mil.txt'.")

# 3. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)
print("Chunks created:", len(docs))

if len(docs) == 0:
    raise ValueError("No chunks created. Check file content or adjust chunk_size.")

print("Sample document content:\n", documents[0].page_content)

# 4. Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# 6. Retriever (limit results to top 2 chunks for relevance)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 7. LLaMA model via Ollama
llm = OllamaLLM(model="tinyllama")

# 8. Prompt template (strict context-only answers)
prompt = ChatPromptTemplate.from_template(
"""
You are a military knowledge assistant.

Answer the question **only using the context below**.  
Do not add information that is not in the context.  
If the answer is not in the context, say "I don't know."  
Always give the specific names or items mentioned in the context, not generic categories.

Context:
{context}

Question:
{question}

Answer:
"""
)
# 9. Format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 10. LCEL chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("🪖 Military AI Assistant Ready (type 'exit' to quit)\n")

# 11. Interactive loop
while True:
    question = input("Ask Question: ")

    if question.lower() == "exit":
        break

    try:
        result = chain.invoke(question)
        print("\nAnswer:", result)
        print()
    except Exception as e:
        print("\n⚠️ Error:", str(e))
        print("Tip: Ensure Ollama is running and the 'tinyllama' model is installed (run `ollama pull tinyllama`).\n")