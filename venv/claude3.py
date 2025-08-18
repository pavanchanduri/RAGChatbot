# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json
import streamlit as st

from botocore.exceptions import ClientError

st.header("Chatbot")

with st.sidebar:
    st.subheader("Input your question below:")
    prompt = st.text_input("Prompt", placeholder="Enter your question here...")

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Format the request payload using the model's native structure.
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

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["content"][0]["text"]
st.subheader("Response:")
st.text_area("Response", value=response_text, height=300)
st.markdown("---")
st.markdown("**Note:** This example uses the native inference API to send a text message to Anthropic Claude. "
            "For more information, see the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/using-native-inference.html).")
st.markdown("**Disclaimer:** This code is for demonstration purposes only and may require additional error handling and validation for production use.")
st.markdown("**Feedback:** If you have any feedback or suggestions, please feel free to reach out or contribute to the project.")
st.markdown("**License:** This code is provided under the Apache License, Version 2.0. See the LICENSE file for details.")