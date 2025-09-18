"""
RAG Preprocessing Script (Pinecone)
===================================

Overview:
---------
This script preprocesses knowledge base sources (S3 text files and web pages), chunks and cleans their content, 
generates embeddings, and stores the resulting vectors in a Pinecone index for retrieval-augmented generation (RAG) chatbots. 
It is designed to run as an AWS Lambda function, triggered by S3 events or scheduled via EventBridge.

Flow Summary:
-------------
1. **Trigger**: Invoked by S3 event (file upload) or scheduled EventBridge rule.
2. **Source Selection**:
    - S3 trigger: Processes all text files in the S3 bucket.
    - Scheduled trigger: Scrapes and processes specified web pages.
3. **Text Loading & Chunking**:
    - Loads text from S3 or scraped web page.
    - Splits text into fixed-size chunks.
    - Cleans each chunk for printable characters and whitespace.
    - Validates chunk encoding.
4. **Embedding Generation**:
    - Uses AWS Bedrock (Cohere embed-english-v3) to generate a vector for each chunk.
    - Retries on throttling or errors.
5. **Vector Upsert to Pinecone**:
    - Chunks and their embeddings are upserted into Pinecone with metadata (chunk text, source).
    - If the index does not exist, it is created with the required dimension and metric.
6. **Completion**:
    - Logs the number of chunks indexed per source.
    - Ready for retrieval by downstream RAG chatbot.

Key Components:
---------------
- **S3 Client**: Loads text files from S3 bucket.
- **BeautifulSoup**: Scrapes and cleans web page content.
- **Chunking & Cleaning**: Splits and sanitizes text for embedding.
- **Bedrock Embeddings**: Generates embedding vectors for each chunk.
- **Pinecone**: Vector database for fast similarity search and upsert.

Detailed Step-by-Step Flow:
--------------------------
1. **Script/Lambda Entry**
    - Entry point is `main(event)` or `lambda_handler(event, context)`.

2. **Trigger Detection**
    - If the event is an S3 trigger, process S3 files.
    - Otherwise, process web pages (scheduled run).

3. **S3 File Processing**
    - List all objects in the S3 bucket.
    - For each `.txt` file:
      - Download and decode text.
      - Split into chunks, clean, and validate.
      - Embed and upsert each chunk into Pinecone.

4. **Web Page Processing**
    - For each URL in the list:
      - Scrape page content (BeautifulSoup, requests).
      - Split into chunks, clean, and validate.
      - Embed and upsert each chunk into Pinecone.

5. **Pinecone Index Management**
    - Checks if the index exists; creates it if not.
    - Index mapping includes vector, chunk text, and source metadata fields.

6. **Embedding and Upsert**
    - For each chunk, generates embedding using Bedrock/Cohere.
    - Upserts chunk and vector into Pinecone with metadata.

7. **Logging and Completion**
    - Prints/logs the number of chunks indexed per source.
    - Indicates completion of preprocessing and upsert.

Environment Variables Required:
------------------------------
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone region

Dependencies:
-------------
- boto3
- requests
- beautifulsoup4
- urllib3
- pinecone-client
- botocore

"""
# This script preprocesses text files from S3, chunks, cleans, embeds, and stores them in Pinecone (vector DB).
#
# AWS EventBridge can be used to schedule this Lambda/script to run at regular intervals (e.g., hourly, daily).
# This ensures the script scrapes and updates the Pinecone index for webpage changes, not just S3 changes.
# Steps: Create an EventBridge rule with a schedule, set the Lambda as the target, and verify scheduled execution.
#


import boto3
import json
import string
import re
import requests
from bs4 import BeautifulSoup
import time
import botocore
import urllib3
import os
from pinecone import Pinecone, ServerlessSpec

# Disable SSL warnings for internal pages with self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Pinecone setup
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pine-api-key")
PINECONE_REGION = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")  # e.g., "us-east-1"
INDEX_NAME = "rag-chatbot-index"
EMBEDDING_DIM = 1024  # Cohere v3 embedding size

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_REGION
        )
    )

index = pc.Index(INDEX_NAME)

s3 = boto3.client("s3")
bucket = "test-bucket-chatbot-321"
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

MAX_CHUNK_LENGTH = 8000

def get_embedding(text, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            body = json.dumps({
                "texts": [text],
                "input_type": "search_document"
            })
            response = bedrock.invoke_model(
                modelId="cohere.embed-english-v3",
                body=body
            )
            return json.loads(response["body"].read())["embeddings"][0]
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == "ThrottlingException":
                print(f"Throttled. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Embedding failed: {e}")
                break
        except Exception as e:
            print(f"Embedding failed: {e}")
            break
    print("Max retries exceeded for embedding.")
    return None

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def clean_chunk(chunk):
    chunk = ''.join(c for c in chunk if c in string.printable)
    chunk = re.sub(r'\s+', ' ', chunk)
    return chunk.strip()

def is_valid_chunk(chunk):
    if not chunk or not chunk.strip():
        return False
    try:
        chunk.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True

def scrape_webpage(url):
    try:
        resp = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""


# --- Trigger type logic ---
import sys
def is_s3_trigger(event=None):
    # S3 event will have 'Records' and 's3' keys
    if event and isinstance(event, dict):
        return 'Records' in event and 's3' in event['Records'][0]
    return False

def main(event=None):
    if is_s3_trigger(event):
        print("S3 trigger detected: Indexing S3 content only.")
        objects = s3.list_objects_v2(Bucket=bucket)
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            if key == "kb_index.json":
                continue
            if key.strip().lower().endswith(".txt"):
                print(f"Processing text file: {key}")
                file_obj = s3.get_object(Bucket=bucket, Key=key)
                text = file_obj["Body"].read().decode("utf-8")
            else:
                continue
            print(f"Extracted text from {key}: {repr(text[:200])}")
            if text.strip():
                for i, chunk in enumerate(chunk_text(text)):
                    chunk = clean_chunk(chunk)
                    if not is_valid_chunk(chunk):
                        continue
                    if len(chunk) > MAX_CHUNK_LENGTH:
                        chunk = chunk[:MAX_CHUNK_LENGTH]
                    try:
                        embedding = get_embedding(chunk)
                        if embedding is not None:
                            vector_id = f"{key}:{i}"
                            index.upsert([
                                {
                                    "id": vector_id,
                                    "values": embedding,
                                    "metadata": {"chunk": chunk, "source": key}
                                }
                            ])
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"Embedding/Pinecone failed for chunk from {key}: {e}")
    else:
        print("Scheduled/EventBridge trigger detected: Indexing web content only.")
        urls = [
            "https://en.wikipedia.org/wiki/Amazon_Web_Services",
            "https://w.amazon.com/bin/view/Transportation/Passport/Passport_QA_Process/"
            # Add more URLs as needed
        ]
        for url in urls:
            print(f"Scraping web page: {url}")
            text = scrape_webpage(url)
            print(f"Extracted text from {url}: {repr(text[:200])}")
            if text.strip():
                for i, chunk in enumerate(chunk_text(text)):
                    chunk = clean_chunk(chunk)
                    if not is_valid_chunk(chunk):
                        continue
                    if len(chunk) > MAX_CHUNK_LENGTH:
                        chunk = chunk[:MAX_CHUNK_LENGTH]
                    try:
                        embedding = get_embedding(chunk)
                        if embedding is not None:
                            vector_id = f"{url}:{i}"
                            index.upsert([
                                {
                                    "id": vector_id,
                                    "values": embedding,
                                    "metadata": {"chunk": chunk, "source": url}
                                }
                            ])
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"Embedding/Pinecone failed for chunk from {url}: {e}")
    print("Preprocessing and Pinecone upsert completed.")

# Lambda handler or script entry point
def lambda_handler(event, context):
    main(event)

if __name__ == "__main__":
    main()