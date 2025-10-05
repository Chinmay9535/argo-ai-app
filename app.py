# app.py

import os
import pandas as pd
import xarray as xr
import xxhash
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
# from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama # Placeholder for a real LLM

# --- Configuration ---
DATA_DIR = os.environ.get('WEBAPP_STORAGE_HOME', '/data')
NC_FILES_PATH = os.path.join(DATA_DIR, "nc_files")
PERSIST_DIRECTORY = os.path.join(DATA_DIR, "argo_vectordb")
PORT = int(os.environ.get('PORT', 8080))

os.makedirs(NC_FILES_PATH, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# --- Initialize Flask App & Load Models ---
app = Flask(__name__)
CORS(app)

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Loading vector store...")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
retriever = vector_store.as_retriever()
print("Models and vector store loaded successfully.")


# --- Processing Function (Your existing automation logic) ---
def process_and_add_files():
    print("Starting file processing...")
    # Load existing IDs from the database
    try:
        existing_ids = set(vector_store.get(include=["metadatas"])['ids'])
        print(f"Found {len(existing_ids)} existing documents.")
    except Exception as e:
        print(f"Could not get existing IDs, starting fresh. Error: {e}")
        existing_ids = set()

    new_documents = []
    # Loop through uploaded files
    for filename in os.listdir(NC_FILES_PATH):
        if filename.endswith(".nc"):
            filepath = os.path.join(NC_FILES_PATH, filename)
            # (Insert your robust file processing logic here from the final Colab script)
            # This is a simplified version for demonstration
            try:
                ds = xr.open_dataset(filepath, decode_times=True)
                df = ds.to_dataframe().reset_index()
                df.dropna(subset=['PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED'], inplace=True)
                
                for _, row in df.iterrows():
                    doc_text = (
                        f"On {row['JULD'].strftime('%Y-%m-%d %H:%M:%S')}, "
                        f"the Argo float with platform number {int(row['PLATFORM_NUMBER'].decode('utf-8').strip())} "
                        f"recorded a temperature of {row['TEMP_ADJUSTED']:.2f} C and salinity of {row['PSAL_ADJUSTED']:.3f} PSU."
                    )
                    doc_id = xxhash.xxh64(doc_text).hexdigest()
                    new_documents.append(Document(page_content=doc_text, metadata={"unique_id": doc_id}))
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")
                continue
    
    # Deduplicate and add to the vector store
    if not new_documents:
        print("No new documents found to process.")
        return 0

    unique_new_docs_map = {doc.metadata["unique_id"]: doc for doc in new_documents}
    docs_to_add = [doc for doc in unique_new_docs_map.values() if doc.metadata["unique_id"] not in existing_ids]

    if docs_to_add:
        print(f"Adding {len(docs_to_add)} new unique documents to the vector store.")
        ids_to_add = [doc.metadata["unique_id"] for doc in docs_to_add]
        vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
        return len(docs_to_add)
    else:
        print("No new unique documents to add.")
        return 0


# --- API Endpoints ---
@app.route("/")
def hello():
    return "Argo AI Backend is live and running on Azure!"

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request."}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files selected for uploading."}), 400

    for file in files:
        if file and file.filename.endswith('.nc'):
            file.save(os.path.join(NC_FILES_PATH, file.filename))
    
    # After saving, trigger the processing function
    num_added = process_and_add_files()
    
    return jsonify({"message": f"Files uploaded successfully. Added {num_added} new records to the database."})


@app.route('/ask', methods=['POST'])
def ask_question():
    # ... (this endpoint remains the same)
    data = request.get_json()
    question = data['question']
    docs = retriever.get_relevant_documents(question)
    if docs:
        answer = docs[0].page_content
    else:
        answer = "Database is empty or no relevant data found. Please upload .nc files first."
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)