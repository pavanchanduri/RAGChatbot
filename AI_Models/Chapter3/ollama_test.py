"""
ollama_test.py - Local LLM Inference with LangChain and Ollama

This script demonstrates how to use a large language model (LLM) locally via the Ollama runtime and the LangChain integration. 
It is designed for production-quality, end-to-end text generation and can be adapted for other prompts, models, or advanced LLM workflows.

Overview:
---------
- Loads a local LLM ("gemma3:4b") using the Ollama runtime and the `langchain_ollama` package.
- Sets the `temperature` parameter to control the creativity/randomness of the generated output.
- Sends a prompt to the model and prints the generated response.

Key Components:
---------------
1. Model Selection and Loading:
	- The `OllamaLLM` class from `langchain_ollama` is used to interface with the Ollama runtime, which manages LLMs on your local machine.
	- The `model` argument specifies the local model to use (e.g., "gemma3:4b").
	- The `temperature` argument controls the randomness of the output (0.0 = deterministic, 1.0+ = highly creative).

2. Local Inference:
	- All inference is performed locally; no data is sent to the cloud.
	- If the specified model is not present, Ollama will attempt to download it to your machine.

3. Prompt and Output:
	- The script sends the prompt "What is an LLM ?" to the model using the `invoke` method.
	- The generated response is printed to the console.

Best Practices:
---------------
- For production use, validate the prompt and handle exceptions during model loading and inference.
- To use a different model, change the `model` argument to a model available in your local Ollama installation.
- Adjust the `temperature` parameter for more or less creative output as needed.
- For batch processing or advanced workflows, wrap the LLM call in a function and handle multiple prompts.

Dependencies:
-------------
- langchain-ollama >= 0.1.0
- ollama (local runtime)
- Python 3.8+

References:
-----------
- LangChain Ollama integration: https://github.com/langchain-ai/langchain-ollama
- Ollama documentation: https://ollama.com/
- Gemma model: https://ollama.com/library/gemma

"""
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:4b", temperature=0.5)
response = llm.invoke("What is an LLM ?")
print(response)