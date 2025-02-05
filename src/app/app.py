import chainlit as cl
from agent import handle_user_query, initialize_component


initialize_component()
@cl.on_message
async def on_message(message: str):
    user_query = message.content  # Extract actual text input
    result = handle_user_query(user_query)  # Pass the text, not the obje
    if isinstance(result, str):
        await cl.Message(content=result).send()
    else:
        response = result["natural_language_response"]
        await cl.Message(content=response).send()

if __name__ == "__main__":
    cl.run()
