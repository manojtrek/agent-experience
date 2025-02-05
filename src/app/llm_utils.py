import json
import logging
import requests
from config import BASE_URL
from langchain.llms import Ollama

llm = Ollama(model="llama3.2", base_url="http://localhost:11434")

def create_llm_prompt(user_query, relevant_endpoints):
    prompt_template = f"""
    You are an API assistant that generates valid API requests based on user queries.
    Follow these guidelines:
    - Return ONLY a single JSON object with no extra text.
    - The JSON object must contain:
      - "endpoint": a string representing the API endpoint. Use ONLY one of the following endpoints:
        {relevant_endpoints} (Choose one and only one endpoint that best matches the user query).
      - "method": one of ["GET", "POST", "PUT", "DELETE"].
      - "parameters": a dictionary of query/body parameters (can be empty).
      - "description": a short summary of the request.
    - Ensure that the selected endpoint accurately reflects the user's intention or query.
    - Generate ONLY ONE API request that best matches the user query.
    - DO NOT generate multiple requests or duplicate endpoints.
    - DO NOT generate the response in array format; it should be a single endpoint.

    Example of a valid API request:
    {{
      "endpoint": "resource",
      "method": "GET",
      "parameters": {{}},
      "description": "Fetches the resource details."
    }}

    If no relevant endpoint is found, return:
    {{
      "endpoint": null,
      "method": null,
      "parameters": {{}},
      "description": "No relevant endpoint found."
    }}

    Now, process the next query and return ONLY the JSON output with no extra text.

    User Query: {user_query}
    """
    return prompt_template

def generate_api_request(user_query, relevant_endpoints):
    if not relevant_endpoints:
        return None

    prompt_template = create_llm_prompt(user_query, relevant_endpoints)  # Get the strict JSON prompt
    formatted_prompt = f"{prompt_template}\nUser Query: {user_query}"  # Append user query manually

    response = llm.invoke(formatted_prompt).strip()  # Invoke the LLM
    print(f"LLM Response: {response}")  # Print the raw response for debugging
    try:
        response_json = response.split("Final Answer:")[-1].strip()
        api_request = json.loads(response_json)
        return api_request  # ✅ Successfully parsed API request
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse API request: {e}")
        return None

def execute_api_request(api_request, api_requester):
    method = api_request.get("method", "").upper()
    endpoint = api_request.get("endpoint", "")
    params = api_request.get("params", {})
    request_body = api_request.get("request_body", {})

    try:
        if method == "GET":
            response = api_requester.get(f"{BASE_URL}{endpoint}", params=params)
        elif method == "POST":
            response = api_requester.post(f"{BASE_URL}{endpoint}", json=request_body)
        elif method == "PUT":
            response = api_requester.put(f"{BASE_URL}{endpoint}", json=request_body)
        elif method == "PATCH":
            response = api_requester.patch(f"{BASE_URL}{endpoint}", json=request_body)
        elif method == "DELETE":
            response = api_requester.delete(f"{BASE_URL}{endpoint}", json=request_body)
        else:
            return "Unsupported HTTP method."

        response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return f"API request failed: {e}"

def generate_natural_language_response(api_response):
    if not api_response or isinstance(api_response, str):
        return "No relevant endpoints found or an error occurred."

    prompt = f"""
    You are an AI assistant that helps users understand API responses. Your task is to summarize the following API response in natural language:

    API Response:
    {json.dumps(api_response, indent=2)}

    Provide a clear and concise summary of the response.
    """
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        logging.error(f"Failed to generate natural language response: {e}")
        return "Failed to generate a natural language response."