"""
RAG Internal Search Chatbot (LangChain + OpenSearch)
====================================================

Overview:
---------
This AWS Lambda function implements a Retrieval-Augmented Generation (RAG) chatbot that answers user queries using a knowledge base indexed in OpenSearch. It leverages LangChain for embedding generation and vector search abstraction, and stores conversation history in DynamoDB.

Flow Summary:
-------------
1. **User Query Input**: Receives a prompt and session_id via an event (API Gateway/Lambda invocation).
2. **Embedding Generation**: Uses AWS Bedrock (Cohere embed-english-v3) to convert the prompt into a vector embedding.
3. **Vector Search (LangChain + OpenSearch)**: Searches OpenSearch for the top-k most similar knowledge base chunks using LangChain's OpenSearchVectorSearch abstraction.
4. **Context Assembly**: Extracts the text content of the retrieved chunks and joins them for context.
5. **Conversation History**: Loads recent conversation history from DynamoDB and formats it for the LLM.
6. **Prompt Construction**: Builds a prompt for the Claude LLM, including conversation history and retrieved context.
7. **LLM Response Generation**: Invokes Claude via Bedrock to generate a response.
8. **History Update**: Saves the new turn (user + bot response) back to DynamoDB.
9. **Response Return**: Returns the bot's answer to the user.

Key Components:
---------------
- **BedrockEmbeddings**: LangChain wrapper for Cohere embedding model on AWS Bedrock.
- **OpenSearchVectorSearch**: LangChain abstraction for vector similarity search in OpenSearch.
- **DynamoDB**: Stores per-session conversation history.
- **Claude LLM (Bedrock)**: Generates final answers using context and history.

Detailed Step-by-Step Flow:
--------------------------
1. **Lambda Handler Entry**
    - Receives `prompt` and `session_id` from the event.
    - Initializes Bedrock client for embedding and LLM calls.

2. **Prompt Embedding**
    - Calls `get_embedding(prompt, bedrock)`.
    - Embedding is a list of 1024 floats representing the semantic meaning of the prompt.

3. **Vector Search in OpenSearch**
    - Initializes LangChain's `OpenSearchVectorSearch` with connection details and embedding model.
    - Calls `similarity_search_by_vector(query_embedding, k=3)` to retrieve top 3 relevant chunks.
    - Each chunk is a document object with `.page_content` (text) and optional metadata.

4. **Context Preparation**
    - Extracts `.page_content` from each document.
    - Joins them with `\n---\n` to form `kb_context`.

5. **Conversation History**
    - Loads last 10 turns from DynamoDB using `get_history(session_id)`.
    - Formats as "User: ...\nBot: ..." for each turn.

6. **Prompt Construction for LLM**
    - Builds a prompt including conversation history, knowledge base context, and the user's question.

7. **Claude LLM Invocation**
    - Sends the constructed prompt to Claude via Bedrock.
    - Receives the bot's response.

8. **History Update**
    - Appends the new turn to history and saves it back to DynamoDB.

9. **Response Return**
    - Returns the bot's answer in the Lambda response.

Environment Variables Required:
------------------------------
- `OPENSEARCH_HOST`: OpenSearch domain endpoint
- `OPENSEARCH_PORT`: OpenSearch port (default 443)
- `OPENSEARCH_USER`: OpenSearch username
- `OPENSEARCH_PASS`: OpenSearch password

Dependencies:
-------------
- boto3
- opensearch-py
- langchain-community

"""
import boto3
import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch

# DynamoDB setup for conversation history
dynamodb = boto3.resource("dynamodb")
history_table = dynamodb.Table("rag_chatbot_history")

def get_history(session_id):
    resp = history_table.get_item(Key={"session_id": session_id})
    return resp.get("Item", {}).get("history", [])

def save_history(session_id, history):
    history_table.put_item(Item={"session_id": session_id, "history": history})


# OpenSearch setup
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST")  # e.g., "search-your-domain.region.es.amazonaws.com"
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "443"))
OPENSEARCH_INDEX = "rag-chatbot-index"
EMBEDDING_DIM = 1024  # Cohere v3 embedding size
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASS")

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

bedrock_embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3", region_name="us-west-2")

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

    # Query OpenSearch for top 3 similar chunks using LangChain
    vector_db = OpenSearchVectorSearch(
        opensearch_url=f"https://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
        index_name=OPENSEARCH_INDEX,
        embedding=bedrock_embeddings,
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        use_ssl=True,
        verify_certs=True,
    )
    docs = vector_db.similarity_search_by_vector(query_embedding, k=3)
    context_chunks = [doc.page_content for doc in docs]
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