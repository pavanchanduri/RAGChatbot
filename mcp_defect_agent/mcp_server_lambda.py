
"""
MCP Defect Agent Lambda Server (LangChain Version)
-------------------------------------------------

Overview:
---------
This AWS Lambda function ingests defect reports from test automation or CI pipelines via API Gateway, deduplicates them, summarizes using a Bedrock LLM (via LangChain), logs unique defects to DynamoDB, and creates JIRA issues for tracking.

End-to-End Flow:
----------------
1. **API Gateway Ingestion**: Receives a JSON payload with test failure details (test_name, error, stack_trace).
2. **Deduplication**: Checks DynamoDB for existing defects with the same test_name and error. If found, returns a duplicate message.
3. **LLM Summarization (LangChain)**: Uses LangChain's Bedrock integration to summarize the defect, producing a title, description, and severity.
4. **DynamoDB Logging**: Stores the unique defect with all details in DynamoDB for analytics and tracking.
5. **JIRA Integration**: Creates a JIRA issue using the Atlassian REST API and Atlassian Document Format for the description.
6. **Response**: Returns a JSON response with defect and JIRA issue details.

Configuration:
--------------
- DynamoDB table name, JIRA credentials, and project key are set via environment variables.
- Bedrock model and region are configurable in the code.

Dependencies:
-------------
- boto3
- requests
- langchain-community

Usage:
------
1. Deploy as an AWS Lambda function behind API Gateway.
2. Set required environment variables for DynamoDB and JIRA.
3. Integrate with test runners or CI/CD pipelines for automated defect management.

"""


import json
import os
import boto3
import requests
from datetime import datetime, timezone
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate

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


def summarize_failure(test_name, error, stack_trace=None):
    """
    Summarize failure using Bedrock LLM via LangChain.
    Constructs a prompt with test failure details and invokes the Bedrock model to generate a defect summary.
    Returns a dict with title, description, and severity.
    """
    print(f"Summarizing failure for test: {test_name}, error: {error}")
    llm = Bedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-west-2"
    )
    prompt_template = PromptTemplate(
        input_variables=["test_name", "error", "stack_trace"],
        template=(
            "A test named '{test_name}' failed.\n"
            "Error: {error}\n"
            "Stack trace: {stack_trace}\n"
            "Please generate a defect summary with a title, description, and severity (High/Medium/Low). "
            "Respond in JSON with keys: title, description, severity."
        )
    )
    prompt = prompt_template.format(
        test_name=test_name,
        error=error,
        stack_trace=stack_trace or ""
    )
    try:
        text = llm(prompt)
        print(f"LLM raw response: {text}")
        import re
        import ast
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