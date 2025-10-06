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
# This path is for Azure's persistent file storage
DATA_DIR = os.environ.get('WEBAPP_STORAGE_HOME', '/data') 
NC_FILES_PATH = os.path.join(DATA_DIR, "nc_files")
PERSIST_DIRECTORY = os.path.join(DATA_DIR, "argo_vectordb")
PORT = int(os.environ.get('PORT', 8080))

# Ensure the directories exist
os.makedirs(NC_FILES_PATH, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# --- Initialize Flask App & Load Models ---
app = Flask(__name__)
CORS(app)

print("DATA PIPELINE: Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("DATA PIPELINE: Loading vector store...")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
print("DATA PIPELINE: Models and vector store loaded successfully.")


def process_and_add_files():
    """
    Scans the upload folder for .nc files, converts them to documents,
    deduplicates them, adds them to the vector store, and cleans up.
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
                    pressure = row['PRES_ADJUSTED']
                    temperature = row['TEMP_ADJUSTED']
                    salinity = row['PSAL_ADJUSTED']
                    latitude = row['LATITUDE']
                    longitude = row['LONGITUDE']

                    doc_text = (
                        f"On {timestamp.strftime('%Y-%m-%d %H:%M:%S')}, the Argo float with platform number {platform_number} "
                        f"(cycle {cycle_number}) recorded a temperature of {temperature:.2f} C and salinity of {salinity:.3f} PSU "
                        f"at a pressure of {pressure:.1f} dbar. The measurement was taken at latitude {latitude:.3f} and longitude {longitude:.3f}."
                    )
                    doc_id = xxhash.xxh64(doc_text).hexdigest()
                    new_documents.append(Document(page_content=doc_text, metadata={"unique_id": doc_id}))
                except Exception as e:
                    print(f"    - WARNING: Skipping a row in {filename} due to a data formatting issue: {e}")
                    continue
            processed_files.append(filepath) 
        except Exception as e:
            print(f"  - ERROR: Could not process file {filename}, skipping it. Reason: {e}")
            continue

    # Deduplicate and add to the vector store
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
    return "Argo AI Data Pipeline is live and running on Azure!"

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files selected for uploading."}), 400

    for file in files:
        if file and file.filename.endswith('.nc'):
            file.save(os.path.join(NC_FILES_PATH, file.filename))
    
    try:
        num_added = process_and_add_files()
        return jsonify({"message": f"Files uploaded. Added {num_added} new records. Check logs for any skipped files."})
    except Exception as e:
        print(f"CRITICAL ERROR during file processing: {e}")
        return jsonify({"error": "A critical error occurred on the server during processing."}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Performs a raw similarity search and returns the most relevant document.
    Useful for testing the contents of the vector database.
    """
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided."}), 400
    
    try:
        # Use the retriever directly to find the most similar raw document
        docs = vector_store.similarity_search(query=question, k=1)
        if docs:
            answer = docs[0].page_content
        else:
            answer = "Database is empty or no relevant documents were found."
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
