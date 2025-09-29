"""
QA.py - Question Answering using HuggingFace Transformers

This script demonstrates how to perform extractive question answering (QA) using a pre-trained transformer model from the HuggingFace Transformers library. 
It is designed for production-quality, end-to-end QA tasks on English text, and can be easily adapted for other languages or models.

Overview:
---------
- Loads a pre-trained DistilBERT model fine-tuned on the SQuAD dataset (Stanford Question Answering Dataset).
- Uses the HuggingFace `pipeline` abstraction for question answering, which handles tokenization, model inference, and answer extraction.
- Accepts a question and a context paragraph, and returns the most probable answer span from the context, along with its confidence score and character offsets.
- Prints the result in a human-readable format using `pprint`.

Key Components:
---------------
1. Model Selection:
    - `distilbert-base-uncased-distilled-squad` is a lightweight, fast, and accurate model for English QA tasks.
    - The model is loaded with both its weights and tokenizer for seamless pipeline integration.

2. Pipeline Initialization:
    - The `pipeline` function abstracts away the details of tokenization, model input formatting, and output post-processing.
    - The `question-answering` task expects two arguments: `question` (the query) and `context` (the passage to search for the answer).

3. Example Usage:
    - The script provides a sample question and context about remote work benefits.
    - The pipeline returns a dictionary with the following keys:
      - `answer`: The extracted answer string from the context.
      - `score`: The model's confidence in the answer (float, 0 to 1).
      - `start`: The character index in the context where the answer starts.
      - `end`: The character index in the context where the answer ends.

4. Output:
    - The result is printed using `pprint` for readability.
    - Example output:
      {'answer': 'flexibility and a better work-life balance',
        'end': 86,
        'score': 0.456,
        'start': 51}

Best Practices:
---------------
- For production use, wrap the pipeline call in a function and add error handling for empty or malformed inputs.
- To use a different model or language, change the `MODEL_NAME` to a compatible QA model from HuggingFace Hub.
- For batch processing, consider using the pipeline in a loop or with a list of questions/contexts.
- For large-scale or real-time applications, load the pipeline once and reuse it to avoid repeated model loading overhead.

Dependencies:
-------------
- transformers >= 4.x
- torch (PyTorch backend)
- pprint (for pretty-printing results)

References:
-----------
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.QuestionAnsweringPipeline
- SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/
- Pre-trained model: https://huggingface.co/distilbert-base-uncased-distilled-squad
"""
from transformers import pipeline
from pprint import pprint

MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# Get predictions
nlp = pipeline(task='question-answering', model=MODEL_NAME, tokenizer=MODEL_NAME)
question = 'What are the benefits of remote work?'
context = (
    'Remote work allows employees to work from anywhere, providing '
    'flexibility and a better work-life balance. It reduces commuting time, '
    'lowers operational costs for companies, and can increase productivity for self-motivated workers.'
)

res = nlp(question=question, context=context)
pprint(res)