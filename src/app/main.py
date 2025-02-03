import json
import os
import requests
import yaml
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document  # Import Document
from langchain.vectorstores.utils import filter_complex_metadata  # Import the utility
from langchain.schema import OutputParserException


base_url = "http://127.0.0.1:8000"  # Define base_url here

# Define the API Requester class
class APIRequester:
    def get(self, url, params=None, headers=None):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in GET request: {e}")

    def post(self, url, json=None, headers=None):
        try:
            response = requests.post(url, json=json, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in POST request: {e}")

    def put(self, url, json=None, headers=None):
        try:
            response = requests.put(url, json=json, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in PUT request: {e}")

    def patch(self, url, json=None, headers=None):
        try:
            response = requests.patch(url, json=json, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in PATCH request: {e}")

    def delete(self, url, json=None, headers=None):
        try:
            response = requests.delete(url, json=json, headers=headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error in DELETE request: {e}")

def download_openapi_spec(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if url.endswith(".yaml") or url.endswith(".yml"):
            return yaml.safe_load(response.text)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading OpenAPI spec: {e}")
        return None

def create_vector_database(openapi_spec_url):
    embeddings = OllamaEmbeddings(model="llama3.2", base_url="http://localhost:11434")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    
    spec = download_openapi_spec(openapi_spec_url)
    if not spec:
        return None

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            endpoint_info = {
                "path": path,
                "method": method.upper(),
                "summary": details.get("summary", "No description"),
                "parameters": details.get("parameters", []),
                "request_body": details.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {}),
                "description": details.get("description", "")
            }
            metadata = {
                "endpoint_path": path,
                "http_method": method.upper(),
                "tags": details.get("tags", [])  # Tags is a list
            }
            # Clean up metadata
            cleaned_metadata = {k: ", ".join(v) if isinstance(v, list) else v or "" for k, v in metadata.items()}
            
            document = Document(page_content=json.dumps(endpoint_info), metadata=cleaned_metadata)
            vectorstore.add_documents([document])
    
    return vectorstore

def create_vector_search_tool(vectorstore):
    def search_function(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=1)
        # Parse each JSON string in the search results into a dictionary
        parsed_results = [json.loads(doc.page_content) for doc in docs]
        return json.dumps(parsed_results)  # Return as a JSON string

    search_tool = Tool(
        name="API_Endpoint_Search",
        func=search_function,
        description="Useful for searching relevant API endpoints based on the user's query."
    )
    return search_tool

def create_llm_prompt(user_query, relevant_endpoints):
    """
    Generates a prompt for the LLM to generate a single API request.
    Args:
        user_query (str): The user's query.
        relevant_endpoints (str): JSON string of relevant endpoints.
    Returns:
        str: The formatted prompt.
    """
    prompt_template = f"""
    You are an API assistant that generates valid API requests based on user queries. 
    Follow these guidelines:
    - Return ONLY a single JSON object with no extra text.
    - The JSON object must contain:
      - "endpoint": a string representing the API endpoint. Use ONLY one of the following endpoints:
        {relevant_endpoints} (Choose one and only one endpoint).
      - "method": one of ["GET", "POST", "PUT", "DELETE"].
      - "parameters": a dictionary of query/body parameters (can be empty).
      - "description": a short summary of the request.
    - Generate ONLY ONE API request that best matches the user query.
    - DO NOT generate multiple requests or duplicate endpoints.
    - DO NOT generate the response in array format, should be single endpoint.

    Now, process the next query and return ONLY the JSON output with no extra text.

    User Query: {user_query}
    """
    return prompt_template


def generate_api_request(llm, user_query, relevant_endpoints):
    prompt_template = create_llm_prompt(user_query, relevant_endpoints)  # Get the strict JSON prompt
    formatted_prompt = f"{prompt_template}\nUser Query: {user_query}"  # Append user query manually

    response = llm.invoke(formatted_prompt).strip()  # Invoke the LLM
    print(f"LLM Response: {response}")  # Print the raw response for debugging
    try:
        response_json = response.split("Final Answer:")[-1].strip()
        api_request = json.loads(response_json)
        return api_request  # ✅ Successfully parsed API request
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: {e}. Retrying...")

    print("❌ LLM failed to generate a valid API request after multiple attempts.")
    return None  


def execute_api_request(api_request, api_requester):
    if not api_request or "endpoint" not in api_request or "method" not in api_request:
        return "Invalid API request format."

    endpoint = api_request["endpoint"]
    method = api_request["method"]
    params = api_request.get("parameters", {})
    request_body = api_request.get("request_body", {})

    try:
        if method == "GET":
            response = api_requester.get(f"{base_url}{endpoint}", params=params)
        elif method == "POST":
            response = api_requester.post(f"{base_url}{endpoint}", json=request_body)
        elif method == "PUT":
            response = api_requester.put(f"{base_url}{endpoint}", json=request_body)
        elif method == "PATCH":
            response = api_requester.patch(f"{base_url}{endpoint}", json=request_body)
        elif method == "DELETE":
            response = api_requester.delete(f"{base_url}{endpoint}", json=request_body)
        else:
            return "Unsupported HTTP method."

        response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
        return response.json()

    except requests.exceptions.RequestException as e:
        return f"API request failed: {e}"  # Return the error message
    
def generate_natural_language_response(llm, api_response):
    prompt = f"""
    You are an AI assistant that helps users understand API responses. Your task is to summarize the following API response in natural language:

    API Response:
    {json.dumps(api_response, indent=2)}

    Provide a clear and concise summary of the response.
    """
    response = llm.invoke(prompt)  # Use `invoke` instead of `__call__`
    return response

def main():
    # Load OpenAPI specifications and create a vector database
    openapi_spec_path = "http://127.0.0.1:8000/openapi.json"  # Change this to the correct path
    vectorstore = create_vector_database(openapi_spec_path)
    
    if not vectorstore:
        print("Failed to initialize vector database.")
        return
    
    # Create a search tool for querying API endpoints
    search_tool = create_vector_search_tool(vectorstore)
    
    # Initialize the LLM (Ollama)
    llm = Ollama(model="llama3.2", base_url="http://localhost:11434")
    
    # Initialize API requester
    api_requester = APIRequester()
    
    while True:
        user_query = input("Ask a question or type 'exit' to quit: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        # Step 1: Retrieve relevant API endpoints
        relevant_endpoints = search_tool.func(user_query)
        print(f"Relevant Endpoints: {relevant_endpoints}")
        
        # Step 2: Generate an API request from user input
        api_request = generate_api_request(llm, user_query, relevant_endpoints)
        if not api_request:
            print("Failed to generate API request.")
            continue
        
        print(f"Generated API Request: {api_request}")
        
        # Step 3: Execute the API request
        api_response = execute_api_request(api_request, api_requester)
        print(f"Raw API Response: {api_response}")
        
        # Step 4: Convert the response into natural language
        natural_language_response = generate_natural_language_response(llm, api_response)
        print(f"Response in Natural Language: {natural_language_response}")

if __name__ == "__main__":
    main()

