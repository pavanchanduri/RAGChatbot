"""
RAG Preprocessing Script (LangChain + OpenSearch)
================================================

Overview:
---------
This script preprocesses knowledge base sources (S3 files: .txt, .pdf, .doc, .docx and web pages), chunks and embeds their content,
and stores the resulting vectors in an OpenSearch index for retrieval-augmented generation (RAG) chatbots.
It is designed to run as an AWS Lambda function, triggered by S3 events or scheduled via EventBridge.


Flow Summary:
-------------
1. **Trigger**: Invoked by S3 event (file upload/update) or scheduled EventBridge rule.
2. **Source Selection**:
    - S3 trigger: Processes only the updated .txt, .pdf, .doc, and .docx files in the S3 bucket (using S3 event data).
    - Scheduled trigger: Scrapes and processes specified web pages.
3. **Change Detection & Efficient Indexing**:
    - For each S3 file, retrieves its ETag (content hash).
    - Checks OpenSearch for existing chunks with the same filename and ETag.
    - If already indexed, skips re-indexing.
    - If changed, deletes old chunks for the file and indexes new ones, storing ETag in metadata.
4. **Text Loading & Chunking**:
    - Loads text from S3 (.txt, .pdf, .doc, .docx) or scraped web page.
    - Splits text into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
    - Each chunk is tagged with metadata (source filename or URL, and ETag for S3 files).
5. **Embedding Generation**:
    - Uses LangChain's `BedrockEmbeddings` (Cohere embed-english-v3 via AWS Bedrock) to generate a vector for each chunk.
6. **Vector Upsert to OpenSearch**:
    - Chunks and their embeddings are upserted into OpenSearch using LangChain's `OpenSearchVectorSearch` abstraction.
    - If the index does not exist, it is created with k-NN enabled for vector search.
7. **Completion**:
    - Logs the number of chunks indexed per source.
    - Ready for retrieval by downstream RAG chatbot.

Key Components:
---------------
- **TextLoader**: Loads text files.
- **PDFLoader**: Loads PDF files.
- **UnstructuredWordDocumentLoader**: Loads .doc and .docx files.
- **RecursiveCharacterTextSplitter**: Splits text into manageable, overlapping chunks for embedding.
- **BedrockEmbeddings**: LangChain wrapper for Cohere embedding model on AWS Bedrock.
- **OpenSearchVectorSearch**: LangChain abstraction for vector similarity search and upsert in OpenSearch.
- **OpenSearch**: AWS-managed vector database for fast similarity search.


Detailed Step-by-Step Flow:
--------------------------
1. **Script/Lambda Entry**
    - Entry point is `main(event)` or `lambda_handler(event, context)`.

2. **Trigger Detection**
    - If the event is an S3 trigger, process only the updated S3 files referenced in the event.
    - Otherwise, process web pages (scheduled run).

3. **S3 File Change Detection & Processing**
    - For each updated S3 file:
      - Retrieve its ETag (content hash).
      - Query OpenSearch for chunks with the same filename and ETag.
      - If found, skip re-indexing.
      - If not found or changed, delete old chunks for the file and index new ones, storing ETag in metadata.
      - Download and decode text (using appropriate loader).
      - Split into chunks with metadata (filename and ETag).
      - Embed and upsert each chunk into OpenSearch.

4. **Web Page Processing**
    - For each URL in the list:
      - Scrape page content (BeautifulSoup, requests).
      - Clean and split into chunks with metadata.
      - Embed and upsert each chunk into OpenSearch.

5. **OpenSearch Index Management**
    - Checks if the index exists; creates it with k-NN enabled if not.
    - Index mapping includes vector, text, source, and ETag metadata fields.

6. **Embedding and Upsert**
    - For each chunk, generates embedding using Bedrock/Cohere.
    - Uses LangChain's `OpenSearchVectorSearch.from_documents` to upsert chunks and vectors.

7. **Logging and Completion**
    - Prints/logs the number of chunks indexed per source.
    - Indicates completion of preprocessing and upsert.

Environment Variables Required:
------------------------------
- `OPENSEARCH_HOST`: OpenSearch domain endpoint
- `OPENSEARCH_PORT`: OpenSearch port (default 443)
- `OPENSEARCH_USER`: OpenSearch username
- `OPENSEARCH_PASS`: OpenSearch password

Dependencies:
-------------
- boto3
- requests
- beautifulsoup4
- urllib3
- langchain-community
- opensearch-py
- PyPDF2
- unstructured
"""
# This script preprocesses text files from S3, chunks, cleans, embeds, and stores them in OpenSearch (vector DB).
#
# AWS EventBridge can be used to schedule this Lambda/script to run at regular intervals (e.g., hourly, daily).
# This ensures the script scrapes and updates the OpenSearch index for webpage changes, not just S3 changes.
# Steps: Create an EventBridge rule with a schedule, set the Lambda as the target, and verify scheduled execution.
# 

# OpenSearch setup instructions:
"""
OpenSearch Setup (AWS Console)
a. Create an OpenSearch Domain
   1. Go to the OpenSearch Service Console.
   2. Click Create domain.
   3. Choose a deployment type (Production/Development).
   4. Set a domain name (e.g., rag-chatbot-domain).
   5. Select an instance type and number of nodes (for dev, t3.small.search is fine).
   6. Enable k-NN under "Network and security" (required for vector search).
   7. Set up access policy:
      For testing, allow your IAM user or Lambda role.
      For production, restrict access to VPC or specific roles.
   8. Click Create and wait for the domain to be active.
b. Get Connection Details
    Find your domain endpoint (e.g., search-rag-chatbot-domain-xxxxxx.region.es.amazonaws.com).
    Note the port (usually 443 for HTTPS).
c. Create Index with k-NN Mapping
"""

import boto3
import os
import requests
from bs4 import BeautifulSoup
import urllib3
from langchain_community.document_loaders import PDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection

# Disable SSL warnings for internal pages with self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# OpenSearch setup
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST")  # e.g., "search-your-domain.region.es.amazonaws.com"
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "443"))
OPENSEARCH_INDEX = "rag-chatbot-index"
EMBEDDING_DIM = 1024  # Cohere v3 embedding size

# If using IAM auth, you may need to use AWS4Auth (see opensearch-py docs)
# For basic auth:
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASS")

# OpenSearch client setup
"""
This client is used to interact with the OpenSearch service.
"""
opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Create index if not exists (with k-NN enabled)
"""
This creates the OpenSearch index with the necessary settings and mappings.
"""
if not opensearch_client.indices.exists(OPENSEARCH_INDEX):
    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM
                },
                "text": {"type": "text"},
                "source": {"type": "keyword"}
            }
        }
    }
    opensearch_client.indices.create(OPENSEARCH_INDEX, body=index_body)

# S3 client creation
s3 = boto3.client("s3")
bucket = "test-bucket-chatbot-321"

# Embedding model setup
# Uses RecursiveCharacterTextSplitter to chunk text into manageable pieces
bedrock_embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3", region_name="us-west-2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Helper functions to load and split text
# Loads text and splits into chunks with metadata
def load_and_split_text(text, source):
    docs = text_splitter.create_documents([text], metadatas=[{"source": source}])
    return docs

# Process S3 files
"""
1. List all objects in the S3 bucket.
2. For each .txt, .pdf, .doc, .docx file:
   - Download and decode text (using appropriate loader).
   - Split into chunks with metadata.
   - Embed and upsert each chunk into OpenSearch.
"""
def process_s3_files(event=None):
    # Determine which files to process and their ETags
    keys_to_process = []
    etags = {}
    if event and 'Records' in event:
        for record in event['Records']:
            s3_info = record['s3']
            key = s3_info['object']['key']
            etag = s3_info['object'].get('eTag')
            keys_to_process.append(key)
            etags[key] = etag
    else:
        objects = s3.list_objects_v2(Bucket=bucket)
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            etag = obj.get("ETag")
            keys_to_process.append(key)
            etags[key] = etag

    for key in keys_to_process:
        if key == "kb_index.json":
            continue
        ext = key.strip().lower().split('.')[-1]
        etag = etags.get(key)
        print(f"Processing file: {key} (ETag: {etag})")
        # Query OpenSearch for existing chunks with this key and ETag
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source": key}},
                        {"term": {"etag": etag}}
                    ]
                }
            }
        }
        results = opensearch_client.search(index=OPENSEARCH_INDEX, body=query)
        if results['hits']['total']['value'] > 0:
            print(f"File {key} with ETag {etag} already indexed. Skipping.")
            continue  # Already indexed, skip

        # Delete old chunks for this key (if any)
        delete_query = {
            "query": {
                "term": {"source": key}
            }
        }
        opensearch_client.delete_by_query(index=OPENSEARCH_INDEX, body=delete_query)

        docs = []
        try:
            file_obj = s3.get_object(Bucket=bucket, Key=key)
            if ext == "txt":
                text = file_obj["Body"].read().decode("utf-8")
                docs = text_splitter.create_documents([text], metadatas=[{"source": key, "etag": etag}])
            elif ext == "pdf":
                with open("/tmp/temp.pdf", "wb") as f:
                    f.write(file_obj["Body"].read())
                loader = PDFLoader("/tmp/temp.pdf")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = key
                    d.metadata["etag"] = etag
            elif ext in ["doc", "docx"]:
                with open(f"/tmp/temp.{ext}", "wb") as f:
                    f.write(file_obj["Body"].read())
                loader = UnstructuredWordDocumentLoader(f"/tmp/temp.{ext}")
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = key
                    d.metadata["etag"] = etag
        except Exception as e:
            print(f"Failed to process {key}: {e}")
            continue
        if docs:
            OpenSearchVectorSearch.from_documents(
                docs,
                bedrock_embeddings,
                opensearch_url=f"https://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
                index_name=OPENSEARCH_INDEX,
                http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
                use_ssl=True,
                verify_certs=True,
            )
            print(f"Indexed {len(docs)} chunks from {key} (ETag: {etag})")

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

def process_webpages():
    urls = [
        "https://en.wikipedia.org/wiki/Amazon_Web_Services",
        "https://w.amazon.com/bin/view/Transportation/Passport/Passport_QA_Process/"
        # Add more URLs as needed
    ]
    for url in urls:
        print(f"Scraping web page: {url}")
        text = scrape_webpage(url)
        docs = load_and_split_text(text, url)
        if docs:
            OpenSearchVectorSearch.from_documents(
                docs,
                bedrock_embeddings,
                opensearch_url=f"https://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
                index_name=OPENSEARCH_INDEX,
                http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
                use_ssl=True,
                verify_certs=True,
            )
        print(f"Indexed {len(docs)} chunks from {url}")

def is_s3_trigger(event=None):
    if event and isinstance(event, dict):
        return 'Records' in event and 's3' in event['Records'][0]
    return False

def main(event=None):
    if is_s3_trigger(event):
        print("S3 trigger detected: Indexing S3 content only.")
        process_s3_files()
    else:
        print("Scheduled/EventBridge trigger detected: Indexing web content only.")
        process_webpages()
    print("Preprocessing and OpenSearch upsert completed.")

# Lambda handler or script entry point
def lambda_handler(event, context):
    main(event)

if __name__ == "__main__":
    main()