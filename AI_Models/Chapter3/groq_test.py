"""
groq_test.py - Cloud LLM Inference with LangChain and Groq

This script demonstrates how to use a large language model (LLM) via the Groq API and the LangChain integration. It is designed for production-quality, end-to-end text generation and can be adapted for other prompts, models, or advanced LLM workflows.

IMPORTANT: You must provide a valid Groq API key to invoke this model. Without an API key, requests to the Groq service will fail.

Overview:
---------
- Loads a cloud-hosted LLM (e.g., 'llama-3.3-70b-versatile') using the Groq API and the `langchain_groq` package.
- Sets the `temperature` parameter to control the creativity/randomness of the generated output.
- Sends a prompt to the model and prints the generated response.

Key Components:
---------------
1. Model Selection and Loading:
    - The `ChatGroq` class from `langchain_groq` is used to interface with the Groq API, which provides access to powerful LLMs in the cloud.
    - The `model_name` argument specifies the model to use (e.g., 'llama-3.3-70b-versatile').
    - The `temperature` argument controls the randomness of the output (0.0 = deterministic, 1.0+ = highly creative).
    - The `api_key` argument (not shown in this example, but required) must be set to your Groq API key for authentication.

2. Cloud Inference:
    - All inference is performed in the cloud; your prompt and data are sent to the Groq API.
    - You must have a valid API key and network access to use this service.

3. Prompt and Output:
    - The script sends the prompt "What is hugging face?" to the model using the `invoke` method.
    - The generated response is printed to the console (using `response.content`).

Best Practices:
---------------
- For production use, securely manage your API key (do not hard-code it in scripts; use environment variables or secret managers).
- To use a different model, change the `model_name` argument to a model available via the Groq API.
- Adjust the `temperature` parameter for more or less creative output as needed.
- For batch processing or advanced workflows, wrap the LLM call in a function and handle multiple prompts.

Dependencies:
-------------
- langchain-groq >= 0.1.0
- Python 3.8+
- A valid Groq API key

References:
-----------
- LangChain Groq integration: https://github.com/langchain-ai/langchain-groq
- Groq API documentation: https://console.groq.com/docs

"""
from langchain_groq import ChatGroq

MODEL_NAME = 'llama-3.3-70b-versatile'
model = ChatGroq(model_name=MODEL_NAME,
                temperature=0.4) # controls creativity

response = model.invoke("What is hugging face?")
print(response.content)