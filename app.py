import os
import pandas as pd
import xarray as xr
import xxhash
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain Imports for vector database functionality
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- Configuration ---
DATA_DIR = os.environ.get('WEBAPP_STORAGE_HOME', '/data')
NC_FILES_PATH = os.path.join(DATA_DIR, "nc_files")
PERSIST_DIRECTORY = os.path.join(DATA_DIR, "argo_vectordb")
PORT = int(os.environ.get('PORT', 8080))

# --- Initialization ---
app = Flask(__name__)
CORS(app)

print("DATA PIPELINE: Loading models and vector store...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
print("DATA PIPELINE: All models loaded successfully.")

def process_and_add_files():
    """
    Scans the upload folder for .nc files, converts them, adds them to the
    vector store, and cleans up. This function includes robust error handling.
    """
    print("Starting file processing...")
    try:
        existing_ids = set(vector_store.get(include=["metadatas"])['ids'])
    except Exception:
        existing_ids = set()

    new_documents = []
    processed_files = []
    files_in_dir = [f for f in os.listdir(NC_FILES_PATH) if f.endswith('.nc')]

    for filename in files_in_dir:
        filepath = os.path.join(NC_FILES_PATH, filename)
        try:
            ds = xr.open_dataset(filepath, decode_times=True)
            df = ds.to_dataframe().reset_index()
            df.dropna(subset=['PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED'], inplace=True)

            for _, row in df.iterrows():
                try:
                    timestamp = row['JULD']
                    platform_number = int(row['PLATFORM_NUMBER'].decode('utf-8').strip())
                    cycle_number = int(row['CYCLE_NUMBER'])
                    # ... add other variables as needed ...
                    doc_text = (
                        f"On {timestamp.strftime('%Y-%m-%d %H:%M:%S')}, the Argo float with platform number {platform_number} "
                        f"(cycle {cycle_number}) recorded a measurement."
                    )
                    doc_id = xxhash.xxh64(doc_text).hexdigest()
                    new_documents.append(Document(page_content=doc_text, metadata={"unique_id": doc_id}))
                except Exception as e:
                    print(f"    - WARNING: Skipping a row in {filename}. Reason: {e}")
                    continue
            processed_files.append(filepath)
        except Exception as e:
            print(f"  - ERROR: Could not process file {filename}, skipping it. Reason: {e}")
            continue

    if not new_documents:
        # Clean up files that were processed but were all duplicates
        for fp in processed_files:
            try:
                os.remove(fp)
            except OSError as e:
                print(f"Error removing processed duplicate file {fp}: {e}")
        return 0

    unique_new_docs_map = {doc.metadata["unique_id"]: doc for doc in new_documents}
    docs_to_add = [doc for doc in unique_new_docs_map.values() if doc.metadata["unique_id"] not in existing_ids]

    if docs_to_add:
        ids_to_add = [doc.metadata["unique_id"] for doc in docs_to_add]
        vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
        num_added = len(docs_to_add)
    else:
        num_added = 0

    # Clean up all successfully processed files
    for fp in processed_files:
        try:
            os.remove(fp)
        except OSError as e:
            print(f"Error removing processed file {fp}: {e}")
            
    return num_added

# --- API Endpoints ---
@app.route("/")
def hello():
    return "ARGO AI Data Pipeline is live!"

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    for file in files:
        if file and file.filename.endswith('.nc'):
            file.save(os.path.join(NC_FILES_PATH, file.filename))
    
    try:
        num_added = process_and_add_files()
        return jsonify({"message": f"Files processed. Added {num_added} new records."})
    except Exception as e:
        print(f"CRITICAL ERROR during file processing: {e}")
        return jsonify({"error": "A critical error occurred on the server."}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Performs a raw similarity search for testing."""
    data = request.get_json()
    question = data.get('question', '')
    try:
        docs = vector_store.similarity_search(query=question, k=1)
        if docs:
            answer = docs[0].page_content
        else:
            answer = "Database is empty or no relevant documents found."
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
