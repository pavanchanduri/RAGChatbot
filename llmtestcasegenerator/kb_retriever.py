
"""
KB Retriever Script for RAG (LangChain + OpenSearch)
===================================================

Overview:
---------
This script provides a function to retrieve relevant knowledge base (KB) context from an OpenSearch vector database
for Retrieval-Augmented Generation (RAG) applications. It is designed to be used as a utility module in LLM-powered
workflows, such as test case generation, where context from similar projects or documents is needed to augment the prompt.

Configuration:
--------------
- OpenSearch connection details are set via environment variables:
    - OPENSEARCH_HOST: OpenSearch domain endpoint
    - OPENSEARCH_PORT: OpenSearch port (default 443)
    - OPENSEARCH_USER: OpenSearch username
    - OPENSEARCH_PASS: OpenSearch password
- OPENSEARCH_INDEX: Name of the index containing embedded KB chunks (should match preprocessing script)

Dependencies:
-------------
- opensearch-py
- langchain-community

Flow Summary:
-------------
1. **Query Embedding**:
    - The input query (e.g., requirements/specification text) is embedded using Bedrock/Cohere via LangChain.
2. **Vector Search**:
    - The query embedding is used to perform a k-NN vector search in the OpenSearch index.
    - The top_k most similar KB chunks are retrieved based on vector similarity.
3. **Context Aggregation**:
    - The text of the retrieved KB chunks is concatenated and returned for use in downstream LLM prompts.

Usage:
------
- Import and call `retrieve_kb_context(query, top_k=5)` from your LLM workflow (e.g., Lambda handler).
- Pass the requirements/specification or user query to retrieve relevant KB context.
- Include the returned context in your prompt to the LLM for improved, context-aware generation.

Integration Example:
--------------------
    from kb_retriever import retrieve_kb_context
    kb_context = retrieve_kb_context(spec_text, top_k=5)
    prompt = f"Spec: {spec_text}\nKB Context: {kb_context}"

"""

import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_community.embeddings import BedrockEmbeddings

# CONFIGURATION (should match RAGPreprocessingScript_Langchain)
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "443"))
OPENSEARCH_INDEX = "rag-project-index"
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASS")

bedrock_embeddings = BedrockEmbeddings(model_id="cohere.embed-english-v3", region_name="us-west-2")

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

def retrieve_kb_context(query, top_k=5):
    """
    Retrieve top_k relevant KB chunks from OpenSearch for the given query using vector search.
    Returns concatenated text of retrieved chunks.
    """
    # Embed the query
    embedding = bedrock_embeddings.embed_query(query)
    # Perform vector search in OpenSearch
    response = opensearch_client.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": embedding,
                        "k": top_k
                    }
                }
            }
        }
    )
    hits = response.get("hits", {}).get("hits", [])
    kb_chunks = [hit["_source"]["text"] for hit in hits if "_source" in hit and "text" in hit["_source"]]
    return "\n---\n".join(kb_chunks)
