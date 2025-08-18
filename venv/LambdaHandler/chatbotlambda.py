import boto3
import json

def lambda_handler(event, context):
    prompt = event.get("prompt", "")
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    try:
        response = client.invoke_model(modelId=model_id, body=request)
        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]
        return {
            "statusCode": 200,
            "body": json.dumps({"response": response_text})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }