"""
Named Entity Recognition (NER) Script using HuggingFace Transformers
===================================================================

Overview:
---------
This script demonstrates how to perform Named Entity Recognition (NER) on a given text using a pre-trained BERT model from the HuggingFace Transformers library. 
NER is a Natural Language Processing (NLP) task that identifies and classifies named entities (such as people, locations, organizations, etc.) in text.

Key Components:
---------------
- **AutoTokenizer**: Automatically loads the appropriate tokenizer for the specified model. The tokenizer splits input text into tokens that the model can process.
- **AutoModelForTokenClassification**: Loads a pre-trained model fine-tuned for token classification tasks like NER.
- **pipeline**: A high-level HuggingFace utility that wraps the model and tokenizer for easy inference.
- **pprint**: Pretty-prints the output for better readability.

Script Steps:
-------------
1. **Import Required Libraries**
	- Import the necessary classes and functions from HuggingFace Transformers and Python's pprint module.

2. **Load Pre-trained Tokenizer and Model**
	- The script loads the tokenizer and model weights for the 'dslim/bert-base-NER' checkpoint. This checkpoint is a BERT model fine-tuned for NER.
	- The tokenizer converts input text into tokens and encodings suitable for the model.
	- The model is loaded for token classification, which means it predicts a label for each token in the input.

3. **Create the NER Pipeline**
	- The pipeline is initialized for the 'ner' task, using the loaded model and tokenizer. This pipeline abstracts away the details of tokenization, model inference, and output formatting.

4. **Prepare Input Text**
	- The variable `text` contains the sentence to be analyzed for named entities.

5. **Run NER Inference**
	- The pipeline is called with the input text. It returns a list of entities found in the text, each with the following information:
	  - `entity`: The predicted entity label (e.g., B-PER for beginning of a person name, I-LOC for inside a location name, etc.)
	  - `score`: The confidence score for the prediction.
	  - `index`: The position of the token in the input.
	  - `start` and `end`: Character positions of the entity in the original text.
	  - `word`: The actual token or word identified as an entity.

6. **Display Results**
	- The results are printed in a readable format using `pprint`.

Notes:
------
- The script may display a warning about unused weights (e.g., 'bert.pooler.dense.weight'). This is expected and safe to ignore, as the pooler layer is not used for token classification.
- The model can recognize standard entity types such as PERSON (PER), ORGANIZATION (ORG), LOCATION (LOC), and MISCELLANEOUS (MISC).
- You can change the `text` variable to analyze different sentences.

Example Output:
---------------
[
  {'entity': 'B-PER', 'score': 0.999, 'index': 4, 'start': 11, 'end': 15, 'word': 'Bert'},
  {'entity': 'B-LOC', 'score': 0.999, 'index': 10, 'start': 29, 'end': 36, 'word': 'Hamburg'}
]

This output indicates that 'Bert' is recognized as a person and 'Hamburg' as a location.
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from pprint import pprint

#%% model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

#%% Named Entity Recognition
text = "My name is Bert and I live in Hamburg"
ner_results = ner_pipeline(text)
pprint(ner_results)