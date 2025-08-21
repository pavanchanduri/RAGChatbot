
import json
import os
import boto3
import requests
from datetime import datetime, timezone

defect_table = boto3.resource('dynamodb').Table(os.environ.get('DEFECT_TABLE', 'mcp_defects'))

# JIRA configuration (set as environment variables)
JIRA_URL = os.environ.get('JIRA_URL', 'https://your-domain.atlassian.net')
JIRA_USER = os.environ.get('JIRA_USER', 'your-email@example.com')
JIRA_API_TOKEN = os.environ.get('JIRA_API_TOKEN', 'your-api-token')
JIRA_PROJECT_KEY = os.environ.get('JIRA_PROJECT_KEY', 'PROJ')


# Bedrock LLM call using anthropic.claude-3-5-sonnet-20240620-v1:0
def summarize_failure(test_name, error, stack_trace=None):
    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
    prompt = (
        f"A test named '{test_name}' failed.\n"
        f"Error: {error}\n"
        f"Stack trace: {stack_trace or ''}\n"
        f"Please generate a defect summary with a title, description, and severity (High/Medium/Low). "
        f"Respond in JSON with keys: title, description, severity."
    )
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=request
        )
        model_response = json.loads(response["body"].read())
        # Try to extract JSON from the model's response
        import re
        import ast
        text = model_response["content"][0]["text"]
        # Extract JSON object from the response text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            summary = json.loads(match.group(0))
        else:
            # Fallback: try to eval if not strict JSON
            summary = ast.literal_eval(text.strip())
        return summary
    except Exception as e:
        # Fallback to default summary if LLM fails
        return {
            'title': f"{test_name} failed: {error[:50]}",
            'description': f"Test '{test_name}' failed with error: {error}. Stack trace: {stack_trace or ''}",
            'severity': 'High' if '500' in error else 'Medium'
        }

def create_jira_issue(summary, description, severity):
    url = f"{JIRA_URL}/rest/api/3/issue"
    auth = (JIRA_USER, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    issue_data = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Bug"},
            # Add custom fields for severity if needed
        }
    }
    response = requests.post(url, auth=auth, headers=headers, data=json.dumps(issue_data))
    if response.status_code == 201:
        return response.json()
    else:
        return {"error": response.text, "status": response.status_code}

def lambda_handler(event, context):
    try:
        body = event.get('body')
        if isinstance(body, str):
            body = json.loads(body)
        test_name = body.get('test_name')
        error = body.get('error')
        stack_trace = body.get('stack_trace')
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check for duplicate defect (same test_name and error)
        # Scan for existing defect with same test_name and error
        existing = defect_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('test_name').eq(test_name) & boto3.dynamodb.conditions.Attr('error').eq(error),
            ProjectionExpression='defect_id'
        )
        if existing.get('Items'):
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Duplicate defect. Not logged again.', 'defect_id': existing['Items'][0]['defect_id']})
            }

        # Summarize defect using LLM
        summary_obj = summarize_failure(test_name, error, stack_trace)
        summary = summary_obj['title']
        description = summary_obj['description']
        severity = summary_obj['severity']

        # Log defect in DynamoDB
        defect_item = {
            'defect_id': f"{test_name}-{timestamp}",
            'test_name': test_name,
            'error': error,
            'stack_trace': stack_trace,
            'summary': summary,
            'description': description,
            'severity': severity,
            'timestamp': timestamp
        }
        defect_table.put_item(Item=defect_item)

        # Create JIRA issue
        jira_result = create_jira_issue(summary, description, severity)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Defect logged', 'defect': defect_item, 'jira_result': jira_result})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
