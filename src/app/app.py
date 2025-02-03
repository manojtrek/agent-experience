from flask import Flask, render_template, request
from api_requester import APIRequester
from vector_search import create_vector_database, create_vector_search_tool
from llm_utils import generate_api_request, execute_api_request, generate_natural_language_response
from config import BASE_URL

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form["query"]

        # Initialize services
        openapi_spec_path = "http://127.0.0.1:8000/openapi.json"
        vectorstore = create_vector_database(openapi_spec_path)
        if not vectorstore:
            return render_template("index.html", message="Failed to initialize vector database.")

        # Create a search tool
        search_tool = create_vector_search_tool(vectorstore)
        relevant_endpoints = search_tool.func(user_query)
        
        print(f"Relevant Endpoints: {relevant_endpoints}")
        
        
        # Step 2: Generate an API request from user input
        api_requester = APIRequester()
        api_request = generate_api_request(user_query, relevant_endpoints)
    
        if not api_request:
            return render_template("index.html", message="Failed to generate API request.")
        
        api_response = execute_api_request(api_request, api_requester)
        natural_language_response = generate_natural_language_response(api_response)

        return render_template("index.html", user_query=user_query, api_request=api_request, api_response=api_response, natural_language_response=natural_language_response)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
