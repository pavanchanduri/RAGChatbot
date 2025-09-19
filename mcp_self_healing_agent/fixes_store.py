"""
Self-Healing Fixes Knowledge Base (Semantic Retrieval)
=====================================================

This module implements a semantic retrieval pipeline for storing and retrieving past self-healing fixes using LangChain and OpenSearch.

End-to-End Flow:
----------------
1. Storing Fixes:
    - When a self-healing fix is generated (e.g., by an LLM), it is stored in OpenSearch as a vector embedding.
    - The fix is saved along with metadata (test name, error, fix, unique ID).
    - Embeddings are generated using Bedrock's Titan model via LangChain.

2. Semantic Retrieval:
    - When a new test failure occurs, the error string is embedded and used to perform a similarity search in OpenSearch.
    - The top-k most similar past fixes are retrieved, enabling context-aware suggestions even for non-exact error matches.

3. Integration:
    - The retrieved past fixes are passed as context to the LLM (via Lambda) to improve the quality of self-healing suggestions.

Key Technologies:
-----------------
- LangChain: Handles embedding generation and vector search abstraction.
- OpenSearch: Stores vector embeddings and supports fast similarity search.
- Bedrock Titan: Embedding model for converting error strings to vectors.

Benefits:
---------
- Enables true Retrieval-Augmented Generation (RAG) for self-healing automation.
- Improves LLM suggestions by leveraging historical fixes semantically.
- Scalable and extensible for large test suites and diverse error types.
"""

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from langchain.docstore.document import Document
import uuid

# OpenSearch configuration
OPENSEARCH_HOST = "https://your-opensearch-domain"  # Replace with your OpenSearch endpoint
OPENSEARCH_INDEX = "self_healing_fixes"

def get_vectorstore():
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-west-2")
    vectorstore = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_HOST,
        index_name=OPENSEARCH_INDEX,
        embedding_function=embeddings.embed_query,
    )
    return vectorstore

def save_fix(test_name, error, fix):
    vectorstore = get_vectorstore()
    doc = Document(
        page_content=error,
        metadata={"test_name": test_name, "fix": fix, "id": str(uuid.uuid4())}
    )
    vectorstore.add_documents([doc])

def get_past_fixes(error, k=3):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(error, k=k)
    return [doc.metadata["fix"] for doc in docs]