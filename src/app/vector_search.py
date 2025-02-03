import os
import json
import requests
import logging
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

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

def create_vector_search_tool(vectorstore):
    def search_function(query: str, score_threshold: float = 0.5) -> str:
        normalized_query = f"Find API matching: {query}"
        
        print(f"\n🔍 Searching for: {normalized_query}")

        # Use `similarity_search_with_score` to check scores
        docs = vectorstore.similarity_search_with_score(normalized_query, k=5)

        # Filter out documents with a score lower than the threshold
        filtered_docs = [
            (doc, score) for doc, score in docs if score >= score_threshold
        ]

        # Print similarity scores
        print("\n📌 Retrieved Documents with Scores:")
        if not filtered_docs:
            print("No relevant results found.")
        else:
            for doc, score in filtered_docs:
                print(f"- {doc.metadata.get('endpoint_path', '')} [{doc.metadata.get('http_method', '')}]")
                print(f"  Score: {score:.4f}")
                print(f"  Summary: {doc.page_content[:100]}...")  # Print first 100 chars

        parsed_results = [
            {
                "endpoint": doc.metadata.get("endpoint_path", ""),
                "method": doc.metadata.get("http_method", ""),
                "summary": doc.page_content,
                "tags": doc.metadata.get("tags", ""),
                "score": score
            }
            for doc, score in filtered_docs
        ]

        return json.dumps(parsed_results, indent=2)

    search_tool = Tool(
        name="API_Endpoint_Search",
        func=search_function,
        description="Useful for searching relevant API endpoints based on the user's query."
    )
    return search_tool
