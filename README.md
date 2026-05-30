# 🪖 Military RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers military-related questions using a custom knowledge base. The application uses FAISS for vector search, Hugging Face embeddings for semantic understanding, Groq LLM for response generation, and Streamlit for the user interface.

---

## 🚀 Features

* Military knowledge-based question answering
* Retrieval-Augmented Generation (RAG)
* FAISS vector database for semantic search
* Hugging Face embeddings
* Groq LLM integration
* Interactive Streamlit chat interface
* Context-aware responses
* Fast document retrieval
* User-friendly UI

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face Embeddings
* Groq API
* Sentence Transformers

---

## 📂 Project Structure

```text
Military-Chatbot/
│
├── rag.py               # RAG pipeline
├── streeam3.py          # Streamlit application
├── mil.txt              # Military knowledge base
├── requirements.txt     # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd Military-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Alternatively, for Streamlit Cloud deployment, add the key in Streamlit Secrets.

---

## ▶️ Run Application

```bash
streamlit run streeam3.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📖 Example Questions

* What are the branches of the Indian Armed Forces?
* Which operation was conducted during the Kargil War?
* Name the fighter jets mentioned in the document.
* What is guerrilla warfare?
* Which missile systems are listed in the document?

---

## 🧠 How It Works

1. Military data is stored in `mil.txt`.
2. Text is split into chunks using LangChain.
3. Embeddings are generated using Hugging Face models.
4. FAISS stores vector embeddings.
5. User questions are converted into embeddings.
6. Relevant document chunks are retrieved.
7. Groq LLM generates answers using retrieved context.
8. Responses are displayed in the Streamlit chat interface.

---

## ☁️ Streamlit Cloud Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select:

```text
streeam3.py
```

as the main file.

5. Add Streamlit Secret:

```toml
GROQ_API_KEY="your_groq_api_key"
```

6. Deploy.

---

## 📌 Future Enhancements

* PDF upload support
* Multiple document support
* Chat history export
* Voice input
* Real-time streaming responses
* User authentication
* Advanced document management

---

## 👨‍💻 Author

Hari Kishor

Computer Science Engineer | Data Analytics | AI & Machine Learning Enthusiast
