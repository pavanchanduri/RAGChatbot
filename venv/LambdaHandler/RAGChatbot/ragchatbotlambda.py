import boto3
import json
import math

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

# DynamoDB setup for conversation history
dynamodb = boto3.resource("dynamodb")
history_table = dynamodb.Table("rag_chatbot_history")

def get_history(session_id):
    resp = history_table.get_item(Key={"session_id": session_id})
    return resp.get("Item", {}).get("history", [])

def save_history(session_id, history):
    history_table.put_item(Item={"session_id": session_id, "history": history})

def lambda_handler(event, context):
    prompt = event.get("prompt", "")
    session_id = event.get("session_id", "default")
    s3 = boto3.client("s3")
    bucket = "test-bucket-chatbot-321"
    region = "us-west-2"
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    # Load KB index from S3
    kb_obj = s3.get_object(Bucket=bucket, Key="kb_index.json")
    kb_index = json.loads(kb_obj["Body"].read().decode("utf-8"))

    # Embed the user prompt using Cohere embed-english-v3
    embed_body = json.dumps({
        "texts": [prompt],
        "input_type": "search_document"
    })
    embed_response = bedrock.invoke_model(
        modelId="cohere.embed-english-v3",
        body=embed_body
    )
    query_embedding = json.loads(embed_response["body"].read())["embeddings"][0]

    # Find top-k most similar chunks
    scored_chunks = []
    for entry in kb_index:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored_chunks.append((score, entry["chunk"]))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for score, chunk in scored_chunks[:3]]  # Top 3

    kb_context = "\n---\n".join(top_chunks)

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