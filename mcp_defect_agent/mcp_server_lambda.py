"""
MCP Defect Agent Lambda Server
-----------------------------
This file implements the AWS Lambda-based server for the MCP Defect Agent.

Purpose:
- Receives defect reports (typically from test automation or CI pipelines) via API Gateway.
- Checks for duplicate defects in DynamoDB to avoid redundant logging.
- Uses AWS Bedrock LLM to summarize the defect and determine severity.
- Logs each unique defect to a DynamoDB table for tracking and analytics.
- Automatically creates a JIRA issue for each new defect using the Atlassian REST API and Atlassian Document Format.

How it works:
- The Lambda handler expects a JSON payload with test failure details (test name, error, stack trace, etc.).
- If the defect is unique, it is summarized, stored in DynamoDB, and logged in JIRA.
- The Lambda returns a JSON response with defect and JIRA issue details.

When to use:
- Deploy this Lambda behind an API Gateway to enable automated, serverless defect logging and JIRA integration for your test automation ecosystem.
- Integrate with test runners or CI/CD pipelines for end-to-end defect management.

Note:
- All configuration (DynamoDB table, JIRA credentials, etc.) is managed via environment variables for security and flexibility.
- For local/server-based (non-Lambda) deployments, see the Flask-based implementation if provided.
"""

import json
import os
import boto3
import requests
from datetime import datetime, timezone

# DynamoDB table name (no sensitive default)
DEFECT_TABLE = os.environ.get('DEFECT_TABLE')
print(f"DEFECT_TABLE: {DEFECT_TABLE}")
defect_table = boto3.resource('dynamodb').Table(DEFECT_TABLE)

# JIRA configuration (no hardcoded defaults)
JIRA_URL = os.environ.get('JIRA_URL')
JIRA_USER = os.environ.get('JIRA_USER')
JIRA_API_TOKEN = os.environ.get('JIRA_API_TOKEN')
JIRA_PROJECT_KEY = os.environ.get('JIRA_PROJECT_KEY')
print(f"JIRA_URL: {JIRA_URL}, JIRA_USER: {JIRA_USER}, JIRA_PROJECT_KEY: {JIRA_PROJECT_KEY}")

"""
Summarize failure using AWS Bedrock LLM
The method constructs a prompt with test failure details and invokes the Bedrock model to generate a defect summary.
"""
def summarize_failure(test_name, error, stack_trace=None):
    print(f"Summarizing failure for test: {test_name}, error: {error}")
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
        import re
        import ast
        text = model_response["content"][0]["text"]
        print(f"LLM raw response: {text}")
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            summary = json.loads(match.group(0))
        else:
            summary = ast.literal_eval(text.strip())
        print(f"LLM summary: {summary}")
        return summary
    except Exception as e:
        print(f"LLM summarization failed: {e}")
        return {
            'title': f"{test_name} failed: {error[:50]}",
            'description': f"Test '{test_name}' failed with error: {error}. Stack trace: {stack_trace or ''}",
            'severity': 'High' if '500' in error else 'Medium'
        }

"""
Create JIRA issue using Atlassian REST API
The method constructs the issue payload in Atlassian Document Format and sends a POST request to create the issue.
"""
def create_jira_issue(summary, description, severity):
    print(f"Creating JIRA issue with summary: {summary}, severity: {severity}")
    url = f"{JIRA_URL}/rest/api/3/issue"
    auth = (JIRA_USER, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    # Atlassian Document Format for description
    description_adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": description}
                ]
            }
        ]
    }
    issue_data = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": description_adf,
            "issuetype": {"name": "Bug"},
        }
    }
    print(f"JIRA payload: {json.dumps(issue_data)}")
    response = requests.post(url, auth=auth, headers=headers, data=json.dumps(issue_data))
    print(f"JIRA response status: {response.status_code}, body: {response.text}")
    if response.status_code == 201:
        return response.json()
    else:
        return {"error": response.text, "status": response.status_code}

"""
AWS Lambda handler for MCP Defect Agent
The method processes incoming defect reports, checks for duplicates, 
summarizes defects, logs them to DynamoDB, and creates JIRA issues.
"""
def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    print(f"Event type: {type(event)}")
    try:
        # Robust body extraction for API Gateway and direct Lambda test
        if isinstance(event, dict) and 'body' in event:
            body = event['body']
        else:
            body = event  # direct invocation or test event

        print(f"Raw body: {body}")
        if isinstance(body, str):
            body = json.loads(body)
        print(f"Parsed body: {body}")

        test_name = body.get('test_name')
        error = body.get('error')
        stack_trace = body.get('stack_trace')
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check for duplicate defect (same test_name and error)
        print(f"Checking for duplicate defect: test_name={test_name}, error={error}")
        existing = defect_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('test_name').eq(test_name) & boto3.dynamodb.conditions.Attr('error').eq(error),
            ProjectionExpression='defect_id'
        )
        print(f"Existing defects: {existing.get('Items')}")
        if existing.get('Items'):
            print("Duplicate defect found, not logging again.")
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
        print(f"Putting defect item in DynamoDB: {defect_item}")
        defect_table.put_item(Item=defect_item)
        print("Defect item put successfully.")

        # Create JIRA issue
        jira_result = create_jira_issue(summary, description, severity)
        print(f"JIRA result: {jira_result}")

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Defect logged', 'defect': defect_item, 'jira_result': jira_result})
        }
    except Exception as e:
        print(f"Exception occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }