

"""
LLM Test Case Generator Lambda (with RAG)
=========================================

Overview:
---------
This AWS Lambda function generates high-level functional and non-functional test cases from user-provided requirements/specifications.
It uses Retrieval-Augmented Generation (RAG) by incorporating relevant context from a knowledge base (KB) of similar projects, indexed in OpenSearch.
The generated test cases are validated by a second LLM call for correctness and completeness.

Flow Summary:
-------------
1. **Input Parsing**:
    - Receives an HTTP event with a document (specification) and filename (supports .txt, .docx, .pdf).
    - Decodes and extracts text from the document.
2. **KB Context Retrieval (RAG)**:
    - Uses the extracted specification as a query to retrieve top-k relevant KB chunks from OpenSearch via `kb_retriever.py`.
    - KB context is concatenated and included in the LLM prompt.
3. **Conversation History**:
    - Loads the last 10 exchanges from DynamoDB for session continuity.
4. **Prompt Construction**:
    - Builds a prompt for the LLM (Claude via Bedrock) including history, specification, and KB context.
5. **Test Case Generation**:
    - Invokes the LLM to generate test cases based on the prompt.
6. **Validation**:
    - Invokes the LLM again to review and validate the generated test cases.
7. **History Update**:
    - Saves the latest user/bot exchange to DynamoDB.
8. **Response**:
    - Returns the generated test cases, validation feedback, and session ID.

Supported Document Formats:
--------------------------
- .txt (plain text)
- .docx (Word, using python-docx)
- .pdf (PDF, using PyPDF2)

RAG Integration:
---------------
- KB context is retrieved from OpenSearch using semantic vector search (via LangChain and Bedrock embeddings).
- The KB is built by a separate preprocessing script that indexes project documents and webpages.

Dependencies:
-------------
- boto3
- python-docx
- PyPDF2
- kb_retriever (utility for OpenSearch KB retrieval)

Usage:
------
1. Deploy as an AWS Lambda function with required dependencies (use a Lambda layer if needed).
2. Ensure OpenSearch KB is indexed and accessible.
3. Configure DynamoDB for conversation history.
4. Send requests with document and filename; receive test cases and validation feedback.

"""
import boto3
import json
import base64
from uuid import uuid4
import os
from kb_retriever import retrieve_kb_context

try:
    import docx
except ImportError:
    docx = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# DynamoDB setup for conversation history
dynamodb = boto3.resource("dynamodb")
history_table = dynamodb.Table("llm_testcasegenerator_history")

def get_history(session_id):
    try:
        resp = history_table.get_item(Key={"session_id": session_id})
        return resp.get("Item", {}).get("history", [])
    except Exception as e:
        print("DYNAMODB GET ERROR:", str(e))
        return []

def save_history(session_id, history):
    try:
        history_table.put_item(Item={"session_id": session_id, "history": history})
    except Exception as e:
        print("DYNAMODB PUT ERROR:", str(e))

def extract_text_from_docx(docx_bytes):
    if not docx:
        return "[python-docx not installed in Lambda layer]"
    from io import BytesIO
    doc = docx.Document(BytesIO(docx_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(pdf_bytes):
    if not PyPDF2:
        return "[PyPDF2 not installed in Lambda layer]"
    from io import BytesIO
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def lambda_handler(event, context):
    print("EVENT:", json.dumps(event))

    # Parse input robustly
    try:
        body = event.get("body", "")
        print("BODY:", body)
        if not body:
            raise ValueError("Empty body")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        data = json.loads(body)
        print("DATA:", data)
    except Exception as e:
        print("INPUT ERROR:", str(e))
        return {
            "statusCode": 400,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Invalid input: {str(e)}"})
        }

    document = data.get("document", "")
    filename = data.get("filename", "")
    session_id = data.get("sessionId") or str(uuid4())
    region = "us-west-2"
    print("Connecting to Bedrock...")
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    if not document:
        print("No document provided")
        return {
            "statusCode": 400,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "No document provided"})
        }


    # Try to extract text based on file extension
    extracted_text = ""
    try:
        print("Extracting text...")
        if filename.endswith(".txt"):
            extracted_text = document
        elif filename.endswith(".docx") or filename.endswith(".pdf"):
            if document.startswith("data:"):
                base64_content = document.split(",", 1)[1]
            else:
                base64_content = document
            file_bytes = base64.b64decode(base64_content)
            if filename.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_bytes)
            else:
                extracted_text = extract_text_from_pdf(file_bytes)
        else:
            extracted_text = document
        print("Extracted text (first 200 chars):", extracted_text[:200])
    except Exception as e:
        print("EXTRACTION ERROR:", str(e))
        extracted_text = f"[Could not extract text from {filename}: {str(e)}]"

    # Retrieve KB context from OpenSearch using the extracted text as query
    try:
        print("Retrieving KB context from OpenSearch...")
        kb_context = retrieve_kb_context(extracted_text, top_k=5)
        print("KB context (first 200 chars):", kb_context[:200])
    except Exception as e:
        print("KB RETRIEVAL ERROR:", str(e))
        kb_context = "[Could not retrieve KB context: {}]".format(str(e))

    # Conversation history (optional, last 10 exchanges)
    MAX_HISTORY = 10
    try:
        print("Getting history...")
        history = get_history(session_id)
        recent_history = history[-MAX_HISTORY:]
    except Exception as e:
        print("DYNAMODB ERROR:", str(e))
        history = []
        recent_history = []

    history_text = ""
    for turn in recent_history:
        history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"


    # Compose prompt for Claude with KB context
    prompt = (
        f"{history_text}"
        f"You are an expert QA engineer. Given the following requirements/specifications and relevant context from similar projects, generate high level functional and non-functional test cases and include some typical edge cases also:\n\n"
        f"Project/Spec:\n{extracted_text}\n\n"
        f"Relevant KB Context:\n{kb_context}"
    )

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8092,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    try:
        print("Invoking Bedrock for test case generation...")
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=request
        )
        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]
        print("Bedrock response (first 200 chars):", response_text[:200])

        # Validation step: Ask Claude to review the generated test cases
        validation_prompt = (
            "You are an expert QA engineer. Review the following test cases for correctness, completeness, and relevance. Point out any issues or improvements.\n\n"
            f"{response_text}"
        )
        validation_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": validation_prompt}],
                }
            ],
        }
        print("Invoking Bedrock for test case validation...")
        validation_response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps(validation_request)
        )
        validation_model_response = json.loads(validation_response["body"].read())
        validation_text = validation_model_response["content"][0]["text"]
        print("Validation feedback (first 200 chars):", validation_text[:200])

        # Update and save conversation history
        history.append({"user": extracted_text, "bot": response_text})
        save_history(session_id, history)

        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "sessionId": session_id,
                "testCases": response_text,
                "validationFeedback": validation_text
            })
        }
    except Exception as e:
        import traceback
        print("BEDROCK ERROR:", str(e))
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }