import os
import pandas as pd
import xarray as xr
import xxhash
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere

# --- Configuration ---
DATA_DIR = os.environ.get('WEBAPP_STORAGE_HOME', '/data')
NC_FILES_PATH = os.path.join(DATA_DIR, "nc_files")
PERSIST_DIRECTORY = os.path.join(DATA_DIR, "argo_vectordb")
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

# --- Initialization ---
app = Flask(__name__)
CORS(app)

print("WEB: Loading models and vector store...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
retriever = vector_store.as_retriever()
llm = ChatCohere(model="command-r", cohere_api_key=COHERE_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
print("WEB: All models loaded successfully.")

def process_single_file(filepath):
    """Processes a single .nc file and adds its unique data to the vector store."""
    try:
        existing_ids = set(vector_store.get(include=["metadatas"])['ids'])
    except Exception:
        existing_ids = set()

    new_documents = []
    try:
        ds = xr.open_dataset(filepath, decode_times=True)
        df = ds.to_dataframe().reset_index()
        df.dropna(subset=['PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED'], inplace=True)

        for _, row in df.iterrows():
            try:
                # ... (your row processing logic) ...
                doc_text = f"Data from {os.path.basename(filepath)}..." # Replace with your full text creation
                doc_id = xxhash.xxh64(doc_text).hexdigest()
                new_documents.append(Document(page_content=doc_text, metadata={"unique_id": doc_id}))
            except Exception as e:
                print(f"    - WARNING: Skipping a row in {os.path.basename(filepath)}. Reason: {e}")
                continue
    except Exception as e:
        print(f"  - ERROR: Could not process file {os.path.basename(filepath)}, skipping. Reason: {e}")
        return 0 # Return 0 documents added

    # Deduplicate and add
    unique_new_docs_map = {doc.metadata["unique_id"]: doc for doc in new_documents}
    docs_to_add = [doc for doc in unique_new_docs_map.values() if doc.metadata["unique_id"] not in existing_ids]

    if docs_to_add:
        ids_to_add = [doc.metadata["unique_id"] for doc in docs_to_add]
        vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
        return len(docs_to_add)
    
    return 0

# --- API Endpoints ---
@app.route("/")
def hello():
    return "Argo AI Backend with Cohere LLM is live!"

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    saved_files = []
    for file in files:
        if file and file.filename.endswith('.nc'):
            filepath = os.path.join(NC_FILES_PATH, file.filename)
            file.save(filepath)
            saved_files.append(filepath)
    
    total_added = 0
    for filepath in saved_files:
        try:
            num_added = process_single_file(filepath)
            total_added += num_added
        finally:
            # CRUCIAL: Delete the file after attempting to process it
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error removing processed file {filepath}: {e}")

    return jsonify({"message": f"Processed {len(saved_files)} files. Added {total_added} new records."})

@app.route('/ask', methods=['POST'])
def ask_question():
    # ... (your existing ask function) ...
    pass
