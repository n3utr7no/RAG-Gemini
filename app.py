import os
import json
import requests
import hashlib
import time
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CHROMA_DB = "chroma_db"
FILES_METADATA = "files_metadata.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

print("Loading embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embedding model loaded successfully!")

def load_files_metadata():
    try:
        if os.path.exists(FILES_METADATA):
            with open(FILES_METADATA, 'r') as f:
                return json.load(f)
        return []
    except:
        return []

def save_files_metadata(metadata):
    try:
        with open(FILES_METADATA, 'w') as f:
            json.dump(metadata, f, indent=2)
    except:
        pass

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def add_file_metadata(filename, original_name, file_hash, file_size):
    metadata = load_files_metadata()
    for item in metadata:
        if item['file_hash'] == file_hash:
            item['last_used'] = time.time()
            save_files_metadata(metadata)
            return metadata

    new_file = {
        'filename': filename,
        'original_name': original_name,
        'file_hash': file_hash,
        'file_size': file_size,
        'upload_time': time.time(),
        'last_used': time.time()
    }

    metadata.append(new_file)
    metadata.sort(key=lambda x: x['upload_time'], reverse=True)

    if len(metadata) > 2:
        files_to_remove = metadata[2:]
        metadata = metadata[:2]
        for old_file in files_to_remove:
            try:
                old_filepath = os.path.join(UPLOAD_FOLDER, old_file['filename'])
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
                try:
                    old_vectordb = Chroma(
                        collection_name=f"collection_{old_file['file_hash']}",
                        embedding_function=embedding,
                        persist_directory=CHROMA_DB
                    )
                    old_vectordb.delete_collection()
                except:
                    pass
            except:
                pass

    save_files_metadata(metadata)
    return metadata

def load_and_process_document(filepath, filename):
    try:
        loader = PyMuPDFLoader(filepath) if filename.lower().endswith(".pdf") else TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(docs), None
    except Exception as e:
        return None, f"Error reading document: {str(e)}"

def get_or_create_vectorstore(texts, collection_name):
    try:
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=CHROMA_DB
        )
        if vectordb._collection.count() == 0:
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embedding,
                collection_name=collection_name,
                persist_directory=CHROMA_DB
            )
        return vectordb, None
    except Exception as e:
        try:
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embedding,
                collection_name=collection_name,
                persist_directory=CHROMA_DB
            )
            return vectordb, None
        except Exception as e2:
            return None, f"Error creating vectorstore: {str(e2)}"

def query_gemini(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]
        return "Sorry, I couldn't generate a response. Please try again."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    except Exception as e:
        return f"Error processing response: {str(e)}"

def process_question_with_file(file_hash, question):
    metadata = load_files_metadata()
    file_info = next((item for item in metadata if item['file_hash'] == file_hash), None)
    if not file_info:
        return "Selected file not found."
    file_info['last_used'] = time.time()
    save_files_metadata(metadata)
    filepath = os.path.join(UPLOAD_FOLDER, file_info['filename'])
    if not os.path.exists(filepath):
        return "File no longer exists on server."
    collection_name = f"collection_{file_hash}"
    try:
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=CHROMA_DB
        )
        if vectordb._collection.count() == 0:
            texts, error = load_and_process_document(filepath, file_info['original_name'])
            if error:
                return error
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embedding,
                collection_name=collection_name,
                persist_directory=CHROMA_DB
            )
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(question)
        if not retrieved_docs:
            return "No relevant information found in the document for your question."
        context = "\n\n".join([f"Passage {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        prompt = f"""You are a helpful assistant that answers questions based on provided context.\nUse only the information from the context below to answer the question. If the answer is not in the context, say \"I don't have enough information to answer this question based on the provided document.\"\n\nContext:\n{context}\n\nQuestion: {question}\n\nPlease provide a clear, accurate answer based on the context above."""
        return query_gemini(prompt)
    except Exception as e:
        return f"Error processing question: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    history = []
    files_metadata = load_files_metadata()
    if request.method == "POST":
        if 'textfile' in request.files and request.files['textfile'].filename:
            file = request.files.get("textfile")
            question = request.form.get("question", "").strip()
            if not question:
                history.append({"question": "", "answer": "Please enter a question."})
                return render_template("index.html", history=history, files=files_metadata)
            filename = file.filename
            timestamp = str(int(time.time()))
            safe_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
            file.save(filepath)
            file_hash = get_file_hash(filepath)
            file_size = os.path.getsize(filepath)
            files_metadata = add_file_metadata(safe_filename, filename, file_hash, file_size)
            texts, error = load_and_process_document(filepath, filename)
            if error:
                history.append({"question": question, "answer": error})
                return render_template("index.html", history=history, files=files_metadata)
            collection_name = f"collection_{file_hash}"
            vectordb, error = get_or_create_vectorstore(texts, collection_name)
            if error:
                history.append({"question": question, "answer": error})
                return render_template("index.html", history=history, files=files_metadata)
            answer = process_question_with_file(file_hash, question)
            history.append({"question": question, "answer": answer, "file_used": filename})
        else:
            selected_file_hash = request.form.get("selected_file")
            question = request.form.get("question", "").strip()
            if not selected_file_hash or not question:
                history.append({"question": question or "", "answer": "Please select a file and enter a question."})
                return render_template("index.html", history=history, files=files_metadata)
            file_name = next((f['original_name'] for f in files_metadata if f['file_hash'] == selected_file_hash), "Unknown File")
            answer = process_question_with_file(selected_file_hash, question)
            history.append({"question": question, "answer": answer, "file_used": file_name})
    return render_template("index.html", history=history, files=files_metadata)

@app.route("/clear")
def clear_history():
    files_metadata = load_files_metadata()
    return render_template("index.html", history=[], files=files_metadata)

@app.route("/api/files")
def get_files():
    return jsonify(load_files_metadata())

if __name__ == "__main__":
    print("Starting Flask RAG application with multi-file support...")
    app.run(debug=True, port=5001, host='127.0.0.1')
