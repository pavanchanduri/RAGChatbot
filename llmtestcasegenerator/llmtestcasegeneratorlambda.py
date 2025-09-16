import boto3
import json
import base64
from uuid import uuid4

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

    # Compose prompt for Claude
    prompt = (
        f"{history_text}"
        f"You are an expert QA engineer. Given the following requirements/specifications, generate high level functional and non-functional test cases and include some typical edge cases also:\n\n"
        f"{extracted_text}"
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