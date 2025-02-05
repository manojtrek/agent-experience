import os
import logging
import requests
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Check if the database already exists
def vectorstore_exists():
    return os.path.exists("db") and os.path.isdir("db")

def download_openapi_spec(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading OpenAPI spec: {e}")
        return None

def create_vector_database(openapi_spec_url):
    # Check if the database already exists, otherwise create it
    if vectorstore_exists():
        logging.info("Vectorstore already exists. Loading the existing vectorstore.")
        vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
        return vectorstore
    
    logging.warning("Vectorstore does not exist. Creating a new one.")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="db")

    spec = download_openapi_spec(openapi_spec_url)
    if not spec:
        return None

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            description = details.get("description", "")
            params = ", ".join([p["name"] for p in details.get("parameters", [])]) if details.get("parameters") else "None"
            
            # Focusing only on the summary and description for embedding
            embedding_text = (
                f"{description}."
            )
            metadata = {
                "endpoint_path": path,
                "http_method": method.upper(),
                "tags": details.get("tags", [])  # Tags is a list
            }
            cleaned_metadata = {k: ", ".join(v) if isinstance(v, list) else v or "" for k, v in metadata.items()}

            document = Document(page_content=embedding_text, metadata=cleaned_metadata)
            vectorstore.add_documents([document])

    logging.info("Vectorstore created and populated.")
    return vectorstore
