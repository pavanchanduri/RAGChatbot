"""
Classifier.py - Zero-Shot Text Classification and Visualization with HuggingFace Transformers

This script demonstrates how to perform zero-shot text classification using a pre-trained transformer model from the HuggingFace Transformers library, 
and how to visualize the classification results using pandas and matplotlib. It is designed for production-quality, end-to-end text classification on English documents, 
and can be easily adapted for other languages, models, or label sets.

Overview:
---------
- Loads a pre-trained BART model (facebook/bart-large-mnli) for zero-shot classification.
- Uses the HuggingFace `pipeline` abstraction for the `zero-shot-classification` task, which handles tokenization, model inference, and output formatting.
- Classifies a set of literary excerpts into user-defined candidate labels (genres: romantic, fantasy, crime) without any task-specific fine-tuning.
- Visualizes the classification scores for a selected document using a color-customized bar plot.

Key Components:
---------------
1. Data Preparation:
    - Three classic English literary excerpts are provided, each representing a different genre (romantic, fantasy, crime).
    - Titles and document texts are stored in lists for easy reference and batch processing.

2. Zero-Shot Classification Pipeline:
    - The `facebook/bart-large-mnli` model is a large, general-purpose English language model fine-tuned for natural language inference (NLI), enabling zero-shot classification.
    - The `pipeline` function abstracts away the details of tokenization, model input formatting, and output post-processing.
    - The `zero-shot-classification` task expects a list of documents and a list of candidate labels.
    - The pipeline returns, for each document, a dictionary with:
      - `sequence`: The input document text.
      - `labels`: The candidate labels, sorted by model confidence.
      - `scores`: The model's confidence for each label (float, 0 to 1).

3. Visualization:
    - The script visualizes the classification scores for the third document ("The Return of Sherlock Holmes") using a bar plot.
    - A pandas DataFrame is created from the result dictionary for easy plotting.
    - Custom bar colors are set for each label to enhance readability.
    - The plot displays the genre labels on the x-axis and their corresponding scores on the y-axis, with the document title as the plot title.

4. Output:
    - The classification results are visualized as a bar chart, showing the model's confidence for each genre label for the selected document.
    - Example output: A bar plot with three bars (romantic, fantasy, crime), each colored differently, and the document title as the plot title.

Best Practices:
---------------
- For production use, validate input documents and candidate labels for length and content.
- To use a different model or language, change the model name in the pipeline to a compatible zero-shot classification model from HuggingFace Hub.
- For batch processing or large datasets, consider parallelizing the pipeline calls and optimizing DataFrame creation.
- For interactive or web-based applications, integrate the visualization into dashboards or reporting tools.

Dependencies:
-------------
- transformers >= 4.x
- torch (PyTorch backend)
- pandas
- matplotlib

References:
-----------
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
- BART-MNLI model: https://huggingface.co/facebook/bart-large-mnli

"""
from transformers import pipeline
import pandas as pd

# %% Data Preparation
# first example: Jane Austen: Pride and Prejudice (romantic novel)
# second example: Lewis Carroll: Alice's Adventures in Wonderland (fantasy novel)
# third example: Arthur Conan Doyle "The Return of Sherlock Holmes" (crime novel)
titles = ["Pride and Prejudice", "Alice's Adventures in Wonderland", "The Return of Sherlock Holmes"]
documents = [
    '''Walt Whitman has somewhere a fine and just distinction between “loving by allowance” and “loving with personal love.” This distinction applies to books as well as to men and women; and in the case of the not very numerous authors who are the objects of the personal affection, it brings a curious consequence with it. There is much more difference as to their best work than in the case of those others who are loved “by allowance” by convention, and because it is felt to be the right and proper thing to love them. And in the sect—fairly large and yet unusually choice—of Austenians or Janites, there would probably be found partisans of the claim to primacy of almost every one of the novels. To some the delightful freshness and humour of Northanger Abbey, its completeness, finish, and entrain, obscure the undoubted critical facts that its scale is small, and its scheme, after all, that of burlesque or parody, a kind in which the first rank is reached with difficulty.''',
    '''Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book, thought Alice “without pictures or conversations? So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.''',
    '''It was in the spring of the year 1894 that all London was interested, and the fashionable world dismayed, by the murder of the Honourable Ronald Adair under most unusual and inexplicable circumstances. The public has already learned those particulars of the crime which came out in the police investigation, but a good deal was suppressed upon that occasion, since the case for the prosecution was so overwhelmingly strong that it was not necessary to bring forward all the facts. Only now, at the end of nearly ten years, am I allowed to supply those missing links which make up the whole of that remarkable chain. The crime was of interest in itself, but that interest was as nothing to me compared to the inconceivable sequel, which afforded me the greatest shock and surprise of any event in my adventurous life. Even now, after this long interval, I find myself thrilling as I think of it, and feeling once more that sudden flood of joy, amazement, and incredulity which utterly submerged my mind. Let me say to that public, which has shown some interest in those glimpses which I have occasionally given them of the thoughts and actions of a very remarkable man, that they are not to blame me if I have not shared my knowledge with them, for I should have considered it my first duty to do so, had I not been barred by a positive prohibition from his own lips, which was only withdrawn upon the third of last month.'''
]

# %% Zero-Shot Text Classification
candidate_labels = ["romantic", "fantasy", "crime"]
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
res = classifier(documents, candidate_labels = candidate_labels)

#%% visualize results
pos = 2
import matplotlib.pyplot as plt
df = pd.DataFrame(res[pos])
ax = df.plot.bar(x='labels', y='scores', title=titles[pos])
bar_colors = ['tab:blue', 'tab:orange', 'tab:green']
for patch, color in zip(ax.patches, bar_colors):
    patch.set_color(color)
plt.show()
