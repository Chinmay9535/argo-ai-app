import os
import pandas as pd
import xarray as xr
import xxhash
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

def process_and_add_files():
    print("Starting file processing...")
    try:
        existing_ids = set(vector_store.get(include=["metadatas"])['ids'])
    except Exception:
        existing_ids = set()

    new_documents = []
    files_in_dir = [f for f in os.listdir(NC_FILES_PATH) if f.endswith('.nc')]

    for filename in files_in_dir:
        filepath = os.path.join(NC_FILES_PATH, filename)
        
        # --- ROBUSTNESS FIX 1: Handle corrupted or unreadable files ---
        try:
            ds = xr.open_dataset(filepath, decode_times=True)
            df = ds.to_dataframe().reset_index()
            df.dropna(subset=['PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED'], inplace=True)

            for _, row in df.iterrows():
                # --- ROBUSTNESS FIX 2: Handle malformed rows ---
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
                except (KeyError, AttributeError, ValueError) as e:
                    print(f"    - WARNING: Skipping a row in {filename} due to a data formatting issue: {e}")
                    continue # Skip to the next row
        except Exception as e:
            print(f"  - ERROR: Could not process file {filename}, skipping it. Reason: {e}")
            continue # Skip to the next file

    # Deduplicate and add to the vector store
    if not new_documents:
        return 0

    unique_new_docs_map = {doc.metadata["unique_id"]: doc for doc in new_documents}
    docs_to_add = [doc for doc in unique_new_docs_map.values() if doc.metadata["unique_id"] not in existing_ids]

    if docs_to_add:
        ids_to_add = [doc.metadata["unique_id"] for doc in docs_to_add]
        vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
        # Clean up processed files
        for filename in files_in_dir:
            os.remove(os.path.join(NC_FILES_PATH, filename))
        return len(docs_to_add)
    else:
        # Clean up processed files even if they are duplicates
        for filename in files_in_dir:
            os.remove(os.path.join(NC_FILES_PATH, filename))
        return 0

# --- API Endpoints ---
@app.route("/")
def hello():
    return "Argo AI Backend with Cohere LLM is live!"

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    for file in files:
        if file and file.filename.endswith('.nc'):
            file.save(os.path.join(NC_FILES_PATH, file.filename))
    
    num_added = process_and_add_files()
    return jsonify({"message": f"Files uploaded. Added {num_added} new records. Check logs for any skipped files."})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    try:
        result = qa_chain.invoke({"query": question})
        return jsonify({"answer": result.get("result", "Could not generate an answer.")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
