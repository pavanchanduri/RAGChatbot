"""
Project Context Preprocessing Script for Agentic RAG
----------------------------------------------------

This script ingests project documents (requirements, policies, specs), existing test cases, and web content, generates vector embeddings using LangChain (BedrockEmbeddings), and indexes them in OpenSearch for semantic retrieval. It supports ETag-based change detection and trigger-based indexing for web sources.

Usage:
------
1. Place project documents (TXT, PDF) and test cases in the data/ directory.
2. Configure web sources in the WEB_SOURCES list.
3. Run this script to preprocess and index project context and web content into OpenSearch.

Dependencies:
-------------
- langchain-community
- opensearch-py
- boto3
- requests

"""

import os
import json
import requests
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch

# OpenSearch configuration
OPENSEARCH_HOST = os.environ.get('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.environ.get('OPENSEARCH_PORT', 9200))
OPENSEARCH_INDEX = os.environ.get('OPENSEARCH_INDEX', 'defect_fixes')

# Bedrock Embeddings configuration
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'amazon.titan-embed-text-v1')
BEDROCK_REGION = os.environ.get('BEDROCK_REGION', 'us-west-2')


# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Web sources to scrape and index (add URLs as needed)
WEB_SOURCES = [
    # Example: {'url': 'https://example.com/project-spec', 'name': 'Project Spec'},
]

# ETag cache file
ETAG_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'etag_cache.json')

# Initialize OpenSearch client
opensearch_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_compress=True,
    http_auth=(os.environ.get('OPENSEARCH_USER'), os.environ.get('OPENSEARCH_PASS')),
    use_ssl=False,
    verify_certs=False
)

# Initialize LangChain Bedrock Embeddings
embeddings = BedrockEmbeddings(
    model_id=BEDROCK_MODEL_ID,
    region_name=BEDROCK_REGION
)

# Initialize LangChain OpenSearch VectorStore
vectorstore = OpenSearchVectorSearch(
    opensearch_url=f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
    index_name=OPENSEARCH_INDEX,
    embedding=embeddings
)

def load_etag_cache():
    if os.path.exists(ETAG_CACHE_FILE):
        with open(ETAG_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_etag_cache(cache):
    with open(ETAG_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def preprocess_and_index_project_context():
    """
    Reads project documents, test cases, and web sources, generates embeddings, and indexes them in OpenSearch.
    Supports ETag-based change detection and trigger-based indexing for web sources.
    """
    # Index local files
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        # Index TXT files (project policies, requirements, test cases)
        if filename.endswith('.txt'):
            with open(filepath, 'r') as f:
                content = f.read()
            metadata = {'filename': filename, 'type': 'document'}
            vectorstore.add_texts([content], metadatas=[metadata])
            print(f"Indexed document: {filename}")
        # Index PDF files (specifications, requirements)
        elif filename.endswith('.pdf'):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                for page in pages:
                    page_content = page.page_content
                    metadata = {'filename': filename, 'type': 'pdf', 'page': page.metadata.get('page', 0)}
                    vectorstore.add_texts([page_content], metadatas=[metadata])
                print(f"Indexed PDF: {filename}")
            except Exception as e:
                print(f"Failed to index PDF {filename}: {e}")
        # Index test cases (JSON format)
        elif filename.endswith('.json') and 'test' in filename.lower():
            with open(filepath, 'r') as f:
                test_cases = json.load(f)
                for tc in test_cases:
                    tc_blob = json.dumps(tc, indent=2)
                    metadata = {'filename': filename, 'type': 'test_case'}
                    vectorstore.add_texts([tc_blob], metadatas=[metadata])
            print(f"Indexed test cases: {filename}")

    # Index web sources with ETag-based change detection
    etag_cache = load_etag_cache()
    for source in WEB_SOURCES:
        url = source.get('url')
        name = source.get('name', url)
        headers = {}
        if url in etag_cache:
            headers['If-None-Match'] = etag_cache[url]
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 304:
                print(f"No change detected for {name} (ETag: {etag_cache[url]})")
                continue
            if response.status_code == 200:
                etag = response.headers.get('ETag')
                if etag:
                    etag_cache[url] = etag
                content = response.text
                metadata = {'url': url, 'name': name, 'type': 'web'}
                vectorstore.add_texts([content], metadatas=[metadata])
                print(f"Indexed web source: {name}")
            else:
                print(f"Failed to fetch {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    save_etag_cache(etag_cache)

if __name__ == "__main__":
    preprocess_and_index_project_context()
    print("Project context preprocessing and indexing complete.")
