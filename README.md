# Argo AI Data Pipeline

A robust, cloud-native data pipeline that automatically ingests Argo float data from `.nc` files, processes it, and stores it in a persistent, searchable vector database. The entire pipeline is served via a Flask API and deployed on Microsoft Azure.

---

## ✨ Features

- **Automated Data Ingestion:** Upload `.nc` files via a simple API endpoint.
- **Robust Processing:** Intelligently handles different data formats (e.g., `JULD` vs. `timestamp`) and gracefully skips corrupted files or malformed data rows without crashing.
- **Vectorization:** Converts scientific data into natural language sentences and stores them as vector embeddings using `sentence-transformers`.
- **Persistent Storage:** Utilizes a persistent ChromaDB vector store, ensuring data is safe across application restarts.
- **Deduplication:** Automatically checks for and ignores duplicate data records, keeping the database clean and efficient.
- **Scalable Deployment:** Containerized with Docker and deployed on Azure App Service for high availability and scalability.

---

## 🛠️ Technology Stack

- **Backend:** Flask
- **Containerization:** Docker
- **Cloud Platform:** Microsoft Azure App Service
- **Storage:** Azure File Share (for persistent data) & Azure Container Registry (for Docker images)
- **Data Processing:** Pandas, Xarray
- **Vector Embeddings & Database:** LangChain, SentenceTransformers, ChromaDB

---

## 🚀 API Endpoints

The application is live and can be accessed at the following base URL:
`http://chinmay-argo-ai-app.azurewebsites.net`

### Health Check

- **Endpoint:** `/`
- **Method:** `GET`
- **Description:** A simple endpoint to confirm that the API is live and running.
- **Success Response (200):**

- "Argo AI Data Pipeline is live!"

- ### Upload `.nc` Files

- **Endpoint:** `/upload`
- **Method:** `POST`
- **Description:** Upload one or more `.nc` files to be processed and added to the vector database. This is a `multipart/form-data` request.
- **Success Response (200):**
```json
{
  "message": "Files processed. Added 3839 new records."
}
