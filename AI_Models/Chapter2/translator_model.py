"""
English-to-Japanese Machine Translation Script using HuggingFace Transformers
============================================================================

Overview:
---------
This script demonstrates how to perform machine translation from English to Japanese using a pre-trained neural translation model from the HuggingFace Transformers library. 
It uses the Mitsua/elan-mt-bt-en-ja model, which is specifically trained for English-to-Japanese translation.

Key Components:
---------------
- **pipeline**: A high-level HuggingFace utility that wraps models and tokenizers for common NLP tasks, including translation.
- **pprint**: Pretty-prints the output for better readability.

Script Steps:
-------------
1. **Import Required Libraries**
	- Import the `pipeline` function from HuggingFace Transformers for translation.
	- Import `pprint` from Python's standard library for pretty-printing results.

2. **Initialize the Translation Pipeline**
	- The pipeline is created for the "translation" task, using the "Mitsua/elan-mt-bt-en-ja" model. This model is trained to translate text from English (en) to Japanese (ja).
	- The pipeline automatically loads the appropriate tokenizer and model weights.

3. **Prepare Input Text**
	- The variable `text` contains the English sentence to be translated into Japanese.

4. **Run Translation**
	- The pipeline is called with the input text. It returns a list of translation results, each containing a dictionary with the key 'translation_text' holding the translated Japanese sentence.

5. **Display Results**
	- The result is printed in a readable format using `pprint`.

Notes:
------
- The translation pipeline for Mitsua/elan-mt-bt-en-ja does not require a `target_lang` argument; it always translates from English to Japanese.
- For other language pairs, use the appropriate model (e.g., Helsinki-NLP/opus-mt-en-fr for English to French).
- The model is based on MarianMT, a neural machine translation architecture.
- The script can be extended to support more languages by initializing different pipelines for each language pair.

Example Output:
---------------
If you run the script, you might see:

	 [{'translation_text': 'あなたが世界で見たい変化になりなさい。'}]

This is the Japanese translation of "Be the change you wish to see in the world."
"""
#%% packages
from transformers import pipeline
from pprint import pprint

#%% model selection
translator = pipeline("translation", "Mitsua/elan-mt-bt-en-ja")

# %%
text = "Be the change you wish to see in the world."
result = translator(text)
pprint(result)