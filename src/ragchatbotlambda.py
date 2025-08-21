import boto3
import json
import os
import requests
from datetime import datetime, timezone
from pinecone import Pinecone

# DynamoDB setup for conversation history
dynamodb = boto3.resource("dynamodb")
history_table = dynamodb.Table("rag_chatbot_history")

def get_history(session_id):
    resp = history_table.get_item(Key={"session_id": session_id})
    return resp.get("Item", {}).get("history", [])

def save_history(session_id, history):
    history_table.put_item(Item={"session_id": session_id, "history": history})

# Pinecone setup
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "rag-chatbot-index"
EMBEDDING_DIM = 1024  # Cohere v3 embedding size

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

MAX_EMBEDDING_LENGTH = 2048  # Cohere v3 embedding model input limit

def get_embedding(prompt, bedrock):
    truncated_prompt = prompt[:MAX_EMBEDDING_LENGTH]
    embed_body = json.dumps({
        "texts": [truncated_prompt],
        "input_type": "search_query"
    })
    embed_response = bedrock.invoke_model(
        modelId="cohere.embed-english-v3",
        body=embed_body
    )
    return json.loads(embed_response["body"].read())["embeddings"][0]

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

def ai_agent_route(prompt, pinecone_matches, score_threshold=0.4):
    """
    Explicit AI agent that decides whether to use internal KB, web, or both.
    Returns a dict: {'route': 'internal'|'web'|'both'|'none', 'context': str}
    """
    internal_chunks = []
    web_chunks = []
    best_internal = None
    best_internal_score = 0

    for match in pinecone_matches:
        score = match.get('score', 0)
        chunk = match['metadata']['chunk']
        source = match['metadata'].get('source', 'internal')
        print(f"Match score: {score}, source: {source}, chunk: {chunk[:100]}")
        if source == "web":
            if score >= score_threshold:
                web_chunks.append(chunk)
        else:
            if score >= score_threshold:
                internal_chunks.append(chunk)
            if score > best_internal_score:
                best_internal = chunk
                best_internal_score = score

    # Agent logic:
    if internal_chunks:
        return {'route': 'internal', 'context': "\n---\n".join(internal_chunks)}
    elif best_internal and best_internal_score > 0.2:
        # If there's a weak internal match, prefer it over web
        return {'route': 'internal', 'context': best_internal}
    elif web_chunks:
        return {'route': 'web', 'context': "\n---\n".join(web_chunks)}
    else:
        return {'route': 'none', 'context': ""}

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

    # Truncate prompt for embedding
    truncated_prompt = prompt[:MAX_EMBEDDING_LENGTH]
    query_embedding = get_embedding(truncated_prompt, bedrock)
    if query_embedding is None:
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Failed to get embedding for prompt"})
        }

    # Query Pinecone for top 3 similar chunks
    result = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # AI agent decides routing
    agent_decision = ai_agent_route(prompt, result['matches'], score_threshold=0.4)
    kb_context = agent_decision['context']
    route = agent_decision['route']

    # If agent says "none", do web search and upsert
    if route == 'none':
        print("AI agent: No relevant context found. Performing Google search...")
        kb_context = google_search(prompt)
        truncated_context = kb_context[:MAX_EMBEDDING_LENGTH]
        new_embedding = get_embedding(truncated_context, bedrock)
        index.upsert([
            {
                "id": f"web_{hash(truncated_context)}",
                "values": new_embedding,
                "metadata": {
                    "chunk": truncated_context,
                    "source": "web",
                    "original_question": prompt,
                    "timestamp": str(datetime.now(timezone.utc))
                }
            }
        ])
        route = 'web'

    # Conversation history
    MAX_HISTORY = 10
    history = get_history(session_id)
    recent_history = history[-MAX_HISTORY:]

    history_text = ""
    for turn in recent_history:
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"

    # Compose prompt for Claude, include routing info for transparency
    full_prompt = (
        f"Conversation so far:\n{history_text}"
        f"Route chosen by AI agent: {route}\n"
        f"Use the following knowledge base context to answer the question.\n"
        f"Knowledge Base Context:\n{kb_context}\n\n"
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
            "body": json.dumps({
                "response": response_text,
                "route": route
            })
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