import json
import boto3
import re
import traceback

def build_prompt(event):
    return f"""
A UI automation test failed.

Test Name: {event.get('test_name', 'N/A')}
Error: {event.get('error', 'N/A')}
Stack Trace:
{event.get('stack_trace', 'N/A')}

DOM Snapshot:
{event.get('dom_snapshot', 'N/A')[:2000]}

Suggest a fix for this failure. If the locator is wrong, suggest a new locator. If it's a timing issue, suggest a wait. Output Python code for the fix.
"""

def extract_new_locator(model_response):
    # Match By.<TYPE>, "<VALUE>" or By.<TYPE>, '<VALUE>' anywhere in the suggestion
    match = re.search(r'By\.(\w+)\s*,\s*["\']([^"\']+)["\']', model_response)
    if match:
        return f'By.{match.group(1)}, "{match.group(2)}"'
    # Try to match presence_of_element_located((By.<TYPE>, "<VALUE>"))
    match = re.search(r'By\.(\w+)\s*,\s*["\']([^"\']+)["\']', model_response)
    if match:
        return f'By.{match.group(1)}, "{match.group(2)}"'
    return None

def lambda_handler(event, context):
    try:
        # Handle API Gateway proxy integration
        if "body" in event and isinstance(event["body"], str):
            event = json.loads(event["body"])
        prompt = build_prompt(event)
        bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        }
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps(request)
        )
        model_response = json.loads(response["body"].read())
        suggestion = model_response["content"][0]["text"]

        print("Model suggestion:", suggestion)  # For debugging

        new_locator = extract_new_locator(suggestion)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "suggestion": suggestion,
                "new_locator": new_locator
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "trace": traceback.format_exc()
            })
        }