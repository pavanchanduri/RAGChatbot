
"""
MCP Self-Healing Selenium Agent Lambda
======================================

End-to-End Flow Overview:
------------------------
1. Test Failure Ingestion:
    - A Selenium test fails and the failure context (test name, error, stack trace, DOM snapshot) is sent to this Lambda function (typically via API Gateway).

2. Retrieval-Augmented Generation (RAG) Pipeline:
    - The Lambda retrieves relevant past fixes for similar errors using semantic search powered by LangChain and OpenSearch.
    - Past fixes are stored as vector embeddings in OpenSearch, enabling context-aware retrieval even for non-exact error matches.

3. Prompt Construction:
    - The failure context and retrieved past fixes are combined into a prompt using LangChain's PromptTemplate.
    - This prompt is designed to guide the LLM (Claude via Bedrock) to suggest the most effective self-healing fix.

4. LLM Invocation:
    - The prompt is sent to the Bedrock Claude model using LangChain's Bedrock integration.
    - The LLM returns a suggestion, which may include Python code, a new locator, or timing adjustments.

5. Locator Extraction:
    - The Lambda parses the LLM's suggestion to extract any new locator information for the failed UI element.

6. Response:
    - The Lambda returns the LLM's suggestion and the extracted locator to the caller (test runner or orchestrator).

7. Knowledge Base Update:
    - If the fix is successful, it can be stored back in the OpenSearch vector database for future semantic retrieval.

Key Technologies:
-----------------
- LangChain: Prompt management, LLM invocation, and semantic retrieval pipeline.
- Bedrock Claude: Large Language Model for generating self-healing suggestions.
- OpenSearch: Vector database for storing and retrieving past fixes using embeddings.
- Selenium: UI automation framework (external to Lambda).

Benefits:
---------
- Context-aware self-healing: Uses semantic search to retrieve relevant past fixes, improving LLM suggestions.
- Modular and extensible: Easily integrates with other vector DBs or LLMs.
- End-to-end automation: Enables closed-loop defect detection and healing in UI automation pipelines.
"""

import json
import re
import traceback
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from mcp_self_healing_agent.fixes_store import get_past_fixes


def build_prompt(event, past_fixes=None):
    prompt_template = PromptTemplate(
        input_variables=["test_name", "error", "stack_trace", "dom_snapshot", "past_fixes"],
        template=(
            "A UI automation test failed.\n\n"
            "Test Name: {test_name}\n"
            "Error: {error}\n"
            "Stack Trace:\n{stack_trace}\n\n"
            "DOM Snapshot:\n{dom_snapshot}\n\n"
            "Past Fixes for Similar Failures:\n{past_fixes}\n\n"
            "Suggest a fix for this failure. If the locator is wrong, suggest a new locator. "
            "If it's a timing issue, suggest a wait. Output Python code for the fix."
        )
    )
    fixes_str = "\n".join(past_fixes) if past_fixes else "None"
    return prompt_template.format(
        test_name=event.get('test_name', 'N/A'),
        error=event.get('error', 'N/A'),
        stack_trace=event.get('stack_trace', 'N/A'),
        dom_snapshot=event.get('dom_snapshot', 'N/A')[:2000],
        past_fixes=fixes_str
    )

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

        # RAG pipeline: retrieve past fixes from KB
        error = event.get("error", "")
        past_fixes = get_past_fixes(error)

        prompt = build_prompt(event, past_fixes)

        llm = Bedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-west-2"
        )
        suggestion = llm(prompt)

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