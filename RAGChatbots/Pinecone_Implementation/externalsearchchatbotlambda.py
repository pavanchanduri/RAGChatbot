"""
External Search Chatbot (Google Search API)
===========================================

Overview:
---------
This AWS Lambda function implements a chatbot that answers user queries using live web search results from the Google Custom Search API. It retrieves relevant snippets, builds context, and uses Claude LLM (via Bedrock) to generate responses. Conversation history is stored in DynamoDB.

Flow Summary:
-------------
1. **User Query Input**: Receives a prompt and session_id via an event (API Gateway/Lambda invocation).
2. **Web Search Context Retrieval**: Uses Google Search API to fetch top 3 relevant snippets for the prompt.
3. **Context Assembly**: Extracts the text content of the retrieved snippets and joins them for context.
4. **Conversation History**: Loads recent conversation history from DynamoDB and formats it for the LLM.
5. **Prompt Construction**: Builds a prompt for the Claude LLM, including conversation history and web search context.
6. **LLM Response Generation**: Invokes Claude via Bedrock to generate a response.
7. **History Update**: Saves the new turn (user + bot response) back to DynamoDB.
8. **Response Return**: Returns the bot's answer to the user.

Key Components:
---------------
- **Google Search API**: Fetches live web search results for user queries.
- **DynamoDB**: Stores per-session conversation history.
- **Claude LLM (Bedrock)**: Generates final answers using context and history.

Detailed Step-by-Step Flow:
--------------------------
1. **Lambda Handler Entry**
    - Receives `prompt` and `session_id` from the event.
    - Initializes Bedrock client for LLM calls.

2. **Web Search Context Retrieval**
    - Calls `google_search(prompt)`.
    - Fetches top 3 snippets using Google Custom Search API.

3. **Context Preparation**
    - Joins snippets with `\n` and truncates to embedding input limit.

4. **Conversation History**
    - Loads last 10 turns from DynamoDB using `get_history(session_id)`.
    - Formats as "User: ...\nBot: ..." for each turn.

5. **Prompt Construction for LLM**
    - Builds a prompt including conversation history, web search context, and the user's question.

6. **Claude LLM Invocation**
    - Sends the constructed prompt to Claude via Bedrock.
    - Receives the bot's response.

7. **History Update**
    - Appends the new turn to history and saves it back to DynamoDB.

8. **Response Return**
    - Returns the bot's answer in the Lambda response.

Environment Variables Required:
------------------------------
- `GOOGLE_API_KEY`: Google Custom Search API key
- `GOOGLE_CSE_ID`: Google Custom Search Engine ID

Dependencies:
-------------
- boto3
- requests

"""
import boto3
import json
import os
import requests
from datetime import datetime, timezone

# DynamoDB setup for conversation history
dynamodb = boto3.resource("dynamodb")
history_table = dynamodb.Table("rag_chatbot_history")

def get_history(session_id):
    resp = history_table.get_item(Key={"session_id": session_id})
    return resp.get("Item", {}).get("history", [])

def save_history(session_id, history):
    history_table.put_item(Item={"session_id": session_id, "history": history})

MAX_EMBEDDING_LENGTH = 2048  # Cohere v3 embedding model input limit

def google_search(query):
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": 3
    }
    try:
        response = requests.get(url, params=params)
        print("Google API response:", response.text)
        if response.status_code != 200:
            return f"Google Search API error: {response.text}"
        results = response.json()
        snippets = []
        for item in results.get("items", []):
            snippets.append(item.get("snippet", ""))
        if not snippets:
            return "No relevant results found on the web."
        return "\n".join(snippets)
    except Exception as e:
        print(f"Google Search Exception: {e}")
        return "Error occurred during Google Search."

def lambda_handler(event, context):
    prompt = event.get("prompt", "")
    session_id = event.get("session_id", "default")
    region = "us-west-2"
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    if not prompt:
        return {
            "statusCode": 400,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "No prompt provided"})
        }

    # Only use web search for context
    kb_context = google_search(prompt)
    truncated_context = kb_context[:MAX_EMBEDDING_LENGTH]

    # Conversation history
    MAX_HISTORY = 10
    history = get_history(session_id)
    recent_history = history[-MAX_HISTORY:]

    history_text = ""
    for turn in recent_history:
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"

    # Compose prompt for Claude
    full_prompt = (
        f"Conversation so far:\n{history_text}"
        f"Use the following web search context to answer the question.\n"
        f"Web Search Context:\n{truncated_context}\n\n"
        f"User Question: {prompt}"
    )

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    try:
        response = bedrock.invoke_model(modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", body=request)
        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]

        # Update and save conversation history
        history.append({"user": prompt, "bot": response_text})
        save_history(session_id, history)

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"response": response_text})
        }
    except Exception as e:
        print(f"Exception in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }