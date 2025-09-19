
# RAG Preprocessing Script (LangChain + OpenSearch)
# Processes project documents from S3 (txt/pdf/docx) and separately scrapes webpages, chunks, embeds, and upserts to OpenSearch

"""
RAG Preprocessing Script (LangChain + OpenSearch)
================================================

Overview:
---------
This script preprocesses project knowledge base sources (S3 documents and web pages), chunks and embeds their content,
and stores the resulting vectors in an OpenSearch index for retrieval-augmented generation (RAG) chatbots.
It is designed to run as an AWS Lambda function, triggered by S3 events or scheduled via EventBridge.

Supported Document Formats:
--------------------------
- .txt (plain text)
- .pdf (PDF, using PyPDF2)
- .docx (Word, using python-docx)

Webpage Support:
---------------
- Webpages are scraped live (not stored in S3) using requests and BeautifulSoup.
- Add URLs to the `urls` list in `process_webpages()`.

Flow Summary:
-------------
1. **Trigger**: Invoked by S3 event (file upload/update) or scheduled EventBridge rule.
2. **Source Selection**:
    - S3 trigger: Processes all supported documents in the S3 bucket under the projects prefix.
    - Scheduled trigger: Scrapes and processes specified web pages.
3. **Text Loading & Chunking**:
    - Loads text from S3 or scraped web page.
    - Splits text into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
    - Each chunk is tagged with metadata (source filename or URL).
4. **Embedding Generation**:
    - Uses LangChain's `BedrockEmbeddings` (Cohere embed-english-v3 via AWS Bedrock) to generate a vector for each chunk.
5. **Vector Upsert to OpenSearch**:
    - Chunks and their embeddings are upserted into OpenSearch using LangChain's `OpenSearchVectorSearch` abstraction.
    - If the index does not exist, it is created with k-NN enabled for vector search.
6. **Completion**:
    - Logs the number of chunks indexed per source.
    - Ready for retrieval by downstream RAG chatbot.

Trigger Details:
---------------
- **S3 Event Trigger**: When a document is uploaded/updated in S3, Lambda is triggered and indexes only S3 content.
- **EventBridge/Scheduled Trigger**: When triggered by EventBridge (e.g., cron schedule), Lambda indexes only webpage content.

Configuration:
--------------
- S3 bucket and prefix are set via `S3_BUCKET` and `PROJECTS_PREFIX`.
- OpenSearch connection details are set via environment variables:
    - `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_USER`, `OPENSEARCH_PASS`
- OpenSearch index name and embedding dimension are configurable.

Dependencies:
-------------
- boto3
- requests
- beautifulsoup4
- python-docx
- PyPDF2
- langchain-community
- opensearch-py

How to Use:
-----------
1. Deploy as an AWS Lambda function with required dependencies (use a Lambda layer if needed).
2. Configure triggers:
    - S3 event for document uploads/updates.
    - EventBridge rule for scheduled webpage indexing.
3. Set environment variables for OpenSearch connection.
4. Add webpage URLs to the `urls` list in `process_webpages()`.
5. Monitor logs for chunk/indexing status.

"""

import boto3
import os
import requests
import json
from io import BytesIO
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection

# CONFIGURATION
S3_BUCKET = "project-info-rag-bucket"  # Change to your S3 bucket name
PROJECTS_PREFIX = "projects/"  # S3 prefix for project documents
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "443"))
OPENSEARCH_INDEX = "rag-project-index"
EMBEDDING_DIM = 1024  # Cohere v3 embedding size
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASS")

s3 = boto3.client("s3")

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Create index if not exists (with k-NN enabled)
if not opensearch_client.indices.exists(OPENSEARCH_INDEX):
    index_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "vector": {"type": "knn_vector", "dimension": EMBEDDING_DIM},
                "text": {"type": "text"},
                "source": {"type": "keyword"}
            }
        }
    }
    opensearch_client.indices.create(OPENSEARCH_INDEX, body=index_body)

bedrock_embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3", region_name="us-west-2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def extract_text_from_txt(txt_bytes):
    return txt_bytes.decode("utf-8")

def extract_text_from_docx(docx_bytes):
    try:
        import docx
    except ImportError:
        return "[python-docx not installed]"
    doc = docx.Document(BytesIO(docx_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(pdf_bytes):
    try:
        import PyPDF2
    except ImportError:
        return "[PyPDF2 not installed]"
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def load_and_split_text(text, source):
    docs = text_splitter.create_documents([text], metadatas=[{"source": source}])
    return docs

def process_s3_files():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=PROJECTS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = os.path.splitext(key)[1].lower()
            file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            file_bytes = file_obj["Body"].read()
            if ext == ".txt":
                text = extract_text_from_txt(file_bytes)
            elif ext == ".pdf":
                text = extract_text_from_pdf(file_bytes)
            elif ext == ".docx":
                text = extract_text_from_docx(file_bytes)
            else:
                continue  # skip unsupported
            docs = load_and_split_text(text, key)
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
            print(f"Indexed {len(docs)} chunks from {key}")

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
        # Add your project-related webpage URLs here
        "https://en.wikipedia.org/wiki/Amazon_Web_Services",
        # ...
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

def lambda_handler(event, context):
    try:
        # S3 event trigger: index S3 content only
        if event and isinstance(event, dict) and 'Records' in event and 's3' in event['Records'][0]:
            print("S3 trigger detected: Indexing S3 content only.")
            process_s3_files()
            msg = "S3 content indexed to OpenSearch."
        # EventBridge/scheduled trigger: index webpage content only
        else:
            print("EventBridge/scheduled trigger detected: Indexing webpage content only.")
            process_webpages()
            msg = "Webpage content indexed to OpenSearch."
        return {
            "statusCode": 200,
            "body": json.dumps({"message": msg})
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
