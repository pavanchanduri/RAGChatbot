import json
import os
from flask import Flask, request, jsonify
import requests
from datetime import datetime

app = Flask(__name__)

# JIRA configuration (set these as environment variables or hardcode for testing)
JIRA_URL = os.environ.get('JIRA_URL', 'https://your-domain.atlassian.net')
JIRA_USER = os.environ.get('JIRA_USER', 'your-email@example.com')
JIRA_API_TOKEN = os.environ.get('JIRA_API_TOKEN', 'your-api-token')
JIRA_PROJECT_KEY = os.environ.get('JIRA_PROJECT_KEY', 'PROJ')


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

@app.route('/log-defect', methods=['POST'])
def log_defect():
    data = request.get_json()
    test_name = data.get('test_name')
    error = data.get('error')
    stack_trace = data.get('stack_trace', '')
    timestamp = datetime.utcnow().isoformat()

    # Simulate LLM summary (replace with Bedrock call in production)
    summary = f"{test_name} failed: {error[:50]}"
    description = f"Test '{test_name}' failed with error: {error}. Stack trace: {stack_trace}"
    severity = 'High' if '500' in error else 'Medium'

    # Create JIRA issue
    jira_result = create_jira_issue(summary, description, severity)

    return jsonify({
        "jira_result": jira_result,
        "defect": {
            "test_name": test_name,
            "error": error,
            "stack_trace": stack_trace,
            "summary": summary,
            "description": description,
            "severity": severity,
            "timestamp": timestamp
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
