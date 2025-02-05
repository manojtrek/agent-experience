import json
import os
import logging
import requests  # Ensure requests is imported
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain.docstore.document import Document  # Ensure Document is imported
from api_requester import APIRequester
from llm_utils import generate_api_request, execute_api_request, generate_natural_language_response
from config import BASE_URL
from langchain.embeddings import HuggingFaceEmbeddings


def direct_llm(query):
    """Uses the LLM directly to answer the query."""
    return llm(query)  # Assuming 'llm' is your initialized LLM

# Load the embedding model
def initialize_component():
    global llm, vectorstore, search_tool_instance, agent,embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = Ollama(model="llama3.2", base_url="http://localhost:11434")  # Corrected model name
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()])

    openapi_spec_path = "http://127.0.0.1:8000/openapi.json"
    vectorstore = create_vector_database(openapi_spec_path)

    # Define the search tool
    search_tool_instance = create_vector_search_tool(vectorstore)
    
    direct_llm_tool = Tool(
        name="DirectLLM",
        func=direct_llm,
        description="Use this tool when the user's query is not related to specific API endpoints or requires general knowledge.  For example, use it for questions like 'What is the weather like?' or 'Tell me a joke.' Do not use this tool for API-related queries."  # Important: Clear description
    )

    tools = [search_tool_instance, direct_llm_tool]  # Add to your tools list
    
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, handle_parsing_errors=True)
    
    print(f"Initialized Tool: {search_tool_instance}")
    print(f"Tools available in agent: {[tool.name for tool in agent.tools]}")



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
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

    spec = download_openapi_spec(openapi_spec_url)
    if not spec:
        return None

    documents = []
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            document = Document(
                page_content=f"{method.upper()} {path} - {operation.get('summary', '')}",
                metadata={
                    "path": path,
                    "method": method,
                    "operation": json.dumps(operation)  # Convert dict to JSON string
                }
            )
            documents.append(document)

    if documents:
        vectorstore.add_documents(documents)
        vectorstore.persist()

    
    return vectorstore

def create_vector_search_tool(vectorstore):
    def search_tool(query):
        print(f"Search query received in tool: {query}")  # Add this line
        results = vectorstore.similarity_search(query, k=1)
        print(f"Search results: {results}")  # Add this line
        if not results:
            return "No relevant endpoints found."
        return results[0].metadata
    return Tool(
        name="VectorSearchTool",
        func=search_tool,
        description="Use this tool to find relevant API endpoints.  Input should be a description of the desired API functionality (e.g., 'get user data', 'create a new product').  The tool returns a list of potentially matching API endpoint descriptions and their metadata.  Look for the 'path' and 'method' in the metadata to construct API requests."
    )

def handle_user_query(user_query):

        print(f"User Query: {user_query}")
        # Step 1: Retrieve relevant API endpoints

        relevant_endpoints = agent.run(user_query)
       # Check if the result has the required API endpoint details
        if not "path:" in relevant_endpoints and "method:" in relevant_endpoints:
            # If not, return the response as is without executing further code.
            return relevant_endpoints
        
        # Step 2: Generate an API request from user input
        api_requester = APIRequester()
        api_request = generate_api_request(user_query, relevant_endpoints)

        if not api_request:
            return "Failed to generate API request."

        # Step 3: Execute the API request
        api_response = execute_api_request(api_request, api_requester)
        print(f"Raw API Response: {api_response}")

        # Step 4: Convert the response into natural language
        natural_language_response = generate_natural_language_response(api_response)
        print(f"Response in Natural Language: {natural_language_response}")

        return {
            "user_query": user_query,
            "api_request": api_request,
            "api_response": api_response,
            "natural_language_response": natural_language_response
        }
