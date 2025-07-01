# RAG-Gemini 

A privacy-focused Retrieval-Augmented Generation (RAG) chatbot using Google Gemini and LangChain, capable of answering questions from user-uploaded documents (PDF or text). Built with Flask, it supports multi-file handling, vector store caching via ChromaDB, and HuggingFace embeddings.

## Features

- **Contextual QA** over uploaded documents using Gemini 1.5 Flash
- **Supports PDF and TXT** file formats
- **HuggingFace Embeddings** (BAAI/bge-small-en-v1.5)
- **ChromaDB** vector store with persistent disk storage
- **Privacy-aware logic** (limited file history, no data logging to Gemini)
- Caches document embeddings; avoids recomputation
- Automatically prunes old uploaded files
- Modern dark-themed web UI with file selection and chat history

## Project Structure

Gemini-RAG-Agent/
├── app.py # Main Flask backend logic
├── templates/
│ └── index.html # Frontend HTML (Jinja2)
├── uploads/ # Folder to store uploaded documents
├── chroma_db/ # Persistent Chroma vectorstore
├── files_metadata.json # Tracks uploaded file metadata
├── requirements.txt # Dependencies
└── README.md


##  Local Deployment

Follow these steps to run the **Gemini-RAG-Agent** on your local machine:

### 1. **Clone the Repository**

git clone https://github.com/n3utr7no/RAG-Gemini.git
cd Gemini-RAG-Agent

### 2. **Install Dependencies**

pip install -r requirements.txt

### 3. **Create a `.env` File**

In the root of the project, create a file named `.env` and add your keys:

GEMINI_API_KEY=your_google_gemini_api_key_here

### 4. **Run the Flask Application**

python app.py

You should see:

 * Running on http://127.0.0.1:5001/ (Press CTRL+C to quit)

### 5. **Access the Web Interface**

Open your browser and visit:
[http://localhost:5001](http://localhost:5001)

Upload a document and ask questions. 


## 📄 License

MIT License

---

Built with 💙 by [@n3utr7no](https://github.com/n3utr7no)


