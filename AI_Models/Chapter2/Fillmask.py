"""
Fillmask.py - Masked Language Modeling with HuggingFace Transformers

This script demonstrates how to use a pre-trained masked language model (MLM) from the HuggingFace Transformers library to perform fill-in-the-blank (fill-mask) predictions. 
It is designed for production-quality, end-to-end masked token inference on English text, and can be easily adapted for other languages or models.

Overview:
---------
- Loads a pre-trained BERT model (bert-base-uncased) for masked language modeling.
- Uses the HuggingFace `pipeline` abstraction for the `fill-mask` task, which handles tokenization, model inference, and output formatting.
- Accepts a sentence with a [MASK] token and predicts the most probable replacements for the masked word.
- Prints the top predictions in a human-readable format using `pprint`.

Key Components:
---------------
1. Model Selection:
	- `bert-base-uncased` is a widely used, general-purpose English language model trained on a large corpus.
	- The model is loaded with its weights and tokenizer for seamless pipeline integration.

2. Pipeline Initialization:
	- The `pipeline` function abstracts away the details of tokenization, model input formatting, and output post-processing.
	- The `fill-mask` task expects input sentences containing a single [MASK] token (case-insensitive).

3. Example Usage:
	- The script provides a sample sentence: "The capital of France is [MASK]."
	- The pipeline returns a list of dictionaries, each representing a possible replacement for the [MASK] token, sorted by model confidence.
	- Each dictionary contains:
	  - `score`: The model's confidence in the prediction (float, 0 to 1).
	  - `token`: The predicted token's integer ID.
	  - `token_str`: The predicted token as a string.
	  - `sequence`: The full sentence with the [MASK] replaced by the predicted token.

4. Output:
	- The result is printed using `pprint` for readability.
	- Example output:
	  [{'score': 0.4167, 'token': 3000, 'token_str': 'paris', 'sequence': 'the capital of france is paris.'},
		{'score': 0.0714, 'token': 22479, 'token_str': 'lille', 'sequence': 'the capital of france is lille.'},
		{'score': 0.0634, 'token': 10241, 'token_str': 'lyon', 'sequence': 'the capital of france is lyon.'}]

Best Practices:
---------------
- For production use, validate that the input contains exactly one [MASK] token.
- To use a different model or language, change the model name in the pipeline to a compatible MLM from HuggingFace Hub.
- For batch processing, use the pipeline in a loop or with a list of masked sentences.
- For large-scale or real-time applications, load the pipeline once and reuse it to avoid repeated model loading overhead.

Dependencies:
-------------
- transformers >= 4.x
- torch (PyTorch backend)
- pprint (for pretty-printing results)

References:
-----------
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.FillMaskPipeline
- BERT model: https://huggingface.co/bert-base-uncased

"""
from transformers import pipeline
from pprint import pprint

unmasker = pipeline("fill-mask", model="bert-base-uncased")

pprint(unmasker("The capital of France is [MASK]."))

# Output: 
# [{'score': 0.4167909026145935, 'token': 3000, 'token_str': 'paris', 'sequence': 'the capital of france is paris.'}, 
# {'score': 0.07141721248626709, 'token': 22479, 'token_str': 'lille', 'sequence': 'the capital of france is lille.'}, 
# {'score': 0.06339294463396072, 'token': 10241, 'token_str': 'lyon', 'sequence': 'the capital of france is lyon.'}, 