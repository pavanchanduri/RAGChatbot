"""
CapstoneProject.py - Multilingual Customer Feedback Classification and Sentiment Analysis

This script demonstrates how to process multilingual customer feedback using HuggingFace Transformers pipelines for 
zero-shot classification and sentiment analysis. It is designed for production-quality, end-to-end feedback analysis, 
and can be adapted for other domains, languages, or label sets.

Overview:
---------
- Loads two pre-trained models:
     - A zero-shot classification model (facebook/bart-large-mnli) for topic/label assignment.
     - A multilingual sentiment analysis model (nlptown/bert-base-multilingual-uncased-sentiment).
- Accepts a list of customer feedback strings in multiple languages (English, German, French).
- Assigns each feedback a sentiment label and the most likely topic label from a candidate set.
- Prints the results in a structured dictionary format.

Key Components:
---------------
1. Data Preparation:
    - The `feedback` list contains three customer feedback examples in English, German, and French, simulating real-world multilingual input.

2. Classification Pipelines:
    - Zero-shot classification uses the BART-MNLI model to assign one of the candidate labels ('defect', 'delivery', 'interface') to each feedback, 
      even if the model was not specifically trained for these topics.
    - Sentiment analysis uses a multilingual BERT model to assign a sentiment label (e.g., 1-5 stars) to each feedback, supporting multiple languages.

3. Processing Function:
    - `process_feedback` takes a list of feedback strings and returns a dictionary with the original feedback, predicted sentiment, and most likely topic label for each entry.
    - The function initializes both pipelines, runs them on the input, and extracts the relevant results.

4. Output:
    - The results are printed as a dictionary with keys: 'feedback', 'sentiment', and 'label'.
    - Each key maps to a list of values corresponding to the input feedback entries.

Best Practices:
---------------
- For production use, validate input data for language and length, and handle exceptions during model loading and inference.
- To use different candidate labels or sentiment models, change the relevant variables in the script.
- For large-scale or real-time applications, consider batching inputs and reusing pipeline objects for efficiency.
- For more readable output, convert the result dictionary to a pandas DataFrame and print it as a table (requires pandas and optionally tabulate).

Dependencies:
-------------
- transformers >= 4.x
- torch (PyTorch backend)
- pprint (for pretty-printing results)

References:
-----------
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers/main_classes/pipelines
- BART-MNLI model: https://huggingface.co/facebook/bart-large-mnli
- Multilingual BERT sentiment model: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

"""

# packages
from transformers import pipeline
from pprint import pprint
from typing import List

# data
feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus."
]

# process feedback
def process_feedback(feedback: List[str]) -> dict[str, List[str]]:
    """
    Process the feedback and return a DataFrame with the sentiment and the most likely label.
    Input:
        feedback: List[str]
    Output:
        pd.DataFrame
    """
    CANDIDATES = ['defect', 'delivery', 'interface']
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
    # initialize the classifiers
    zero_shot_classifier = pipeline(task="zero-shot-classification", 
                                    model=ZERO_SHOT_MODEL)
    sentiment_classifier = pipeline(task="text-classification", 
                                    model=SENTIMENT_MODEL)

    zero_shot_res = zero_shot_classifier(feedback, 
                                         candidate_labels = CANDIDATES)
    sentiment_res = sentiment_classifier(feedback)
    sentiment_labels = [res['label'] for res in sentiment_res]
    most_likely_labels = [res['labels'][0] for res in zero_shot_res]
    res = {'feedback': feedback, 'sentiment': sentiment_labels, 'label': most_likely_labels}
    return res

# Test
result = process_feedback(feedback)
pprint(result)