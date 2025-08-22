import boto3
import json
import os
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

def get_embedding(prompt, bedrock):
    embed_body = json.dumps({
        "texts": [prompt],
        "input_type": "search_query"
    })
    embed_response = bedrock.invoke_model(
        modelId="cohere.embed-english-v3",
        body=embed_body
    )
    return json.loads(embed_response["body"].read())["embeddings"][0]

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

    # Get embedding for the user prompt
    query_embedding = get_embedding(prompt, bedrock)
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

    # Gather the retrieved chunks for context
    context_chunks = []
    for match in result['matches']:
        context_chunks.append(match['metadata']['chunk'])

    kb_context = "\n---\n".join(context_chunks)

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
            "body": json.dumps({"response": response_text})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }