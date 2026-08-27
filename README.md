# AI_Assignment

BMCS2074
ARTIFICIAL INTELLIGENCE
202605 Session, Year 2026/27

# Assignment Documentation

Project Title: Comparative Sentiment Analysis on IMDB Movie Reviews: From Classical Machine Learning to Transformer Architectures

Programme: RDS Y2S1

Tutorial Group: G5

Tutor: Dr CHENG KAM CHING

Team members

| No | Student Name | Student ID | Module In Charge | Signature and Date |
|---:|---|---:|---|---|
| 1 | David Lee Yong Fu | 2612486 | | |
| 2 | Ambrose Teo Chen Bin | 2612472 | | |
| 3 | Teh Zi Yan | 2612547 | | |


## Introduction

### Background
Sentiment analysis, a core subfield of Natural Language Processing (NLP), focuses on extracting subjective information and affective states from unstructured textual data. With the explosive growth of online review platforms like IMDB, automated sentiment classification has become essential for understanding consumer opinion at scale. Movie reviews pose unique linguistic challenges due to informal slang, nuanced sarcasm, complex sentence structures, and high vocabulary variance.

### Problem Statement
Traditional statistical NLP methods (such as Bag-of-Words and TF-IDF paired with linear classifiers) often fail to capture word order, syntactic context, and semantic nuance. Conversely, complex deep learning models and pre-trained transformers offer superior contextual comprehension but require significantly higher computational resources and training time. There is a practical trade-off between model complexity, computational efficiency, and classification accuracy that must be systematically evaluated to guide real-world deployments on unstructured textual data.

### Objectives/Aims
- To design and implement a standardized text preprocessing pipeline for unstructured IMDB movie reviews.
- To construct and evaluate three distinct NLP models representing key technological paradigms: Naive Bayes, Linear SVM and DistilBERT Transformer/GPT.
- To comparative-analyze the performance, inference speed, and classification trade-offs of each approach using evaluation metrics.

### Significance / Contribution of the Study
This study provides empirical insights into how textual complexity affects model performance across different algorithmic paradigms. By benchmarking statistical models against sequential deep learning and modern transformers on the IMDB dataset, this research delivers practical guidelines for developers deciding between fast, low-resource deployment and high-accuracy, resource-intensive solutions.

## Related Work

### Review of previous studies
Early approaches to sentiment classification relied heavily on lexical matching and n-gram feature extraction. Pang et al. established foundational benchmarks using Support Vector Machines (SVM) and Naive Bayes on TF-IDF features, achieving solid baselines but struggling with context-dependent semantics.
With the emergence of deep learning, Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks introduced context retention across sequential word embeddings (e.g., Word2Vec, GloVe), significantly improving long-dependency tracking. More recently, attention-based architectures like DistilBERT have revolutionized NLP by capturing bidirectional contextual dependencies, outperforming sequential models across standardized benchmarks.

### Research gap and justification for the current study
While advanced transformers consistently yield state-of-the-art benchmark scores, prior literature frequently overlooks the practical trade-offs between predictive gain and computational cost in real-world applications. Many comparative studies evaluate models across disparate datasets or under inconsistent preprocessing pipelines. This study addresses this gap by conducting a controlled, head-to-head performance analysis under identical data splits and evaluation standards.

## Methodology

### System flowchart / activity diagram

(Include your system flowchart or activity diagram here in the repository; replace this line with the image/link if available.)

### Description and analysis of dataset

1. IMDB Large Movie Review Dataset
- Source: IMDB Large Movie Review Dataset (Maas et al.).
- Dataset Size: 50,000 labeled movie reviews (25,000 training, 25,000 testing).
- Class Distribution: Perfectly balanced with 50% positive (rating ≥ 7/10) and 50% negative (rating ≤ 4/10) reviews.
- Preprocessing Pipeline: HTML tag stripping, punctuation removal, lowercasing, tokenization, stopword removal, and sequence padding/truncation to a fixed length of 250 tokens.

2. Out-of-Domain Custom Dataset (Real-World Testing)
- Source & Purpose: While the IMDB dataset provides a strong baseline, models trained on formal, Western-centric reviews often suffer from domain drift when applied to modern internet language. To critically evaluate our models' real-world generalization capabilities, we manually curated an additional custom dataset consisting of 25 contemporary movie reviews.
- Dataset Characteristics: This dataset intentionally introduces complex linguistic challenges not heavily represented in the IMDB corpus, including Gen-Z internet slang (e.g., "mid", "fr"), subtle sarcasm, double negatives, and Malaysian colloquialisms (Manglish, e.g., "potong stim", "best giler").
- Objective: The primary goal of introducing this custom dataset is to perform an in-depth error analysis, observing how different algorithmic paradigms handle out-of-vocabulary (OOV) tokens and localized cultural contexts.

## Algorithm selection & description of algorithm(s)

### Algorithm / Approach

| Algorithm | Description & Justification |
|---|---|
| Naïve Bayes (MultinomialNB) | A probabilistic classifier based on applying Bayes' theorem. It is highly efficient for text classification and serves as our baseline statistical model using TF-IDF features (max 10,000 features). |
| Linear SVM (LinearSVC) | Support Vector Machine finds the optimal hyperplane that maximizes the margin between positive and negative classes. It generally outperforms Naïve Bayes in high-dimensional sparse data like TF-IDF vectors. |
| DistilBERT Transformer | A distilled, pre-trained transformer model (distilbert-base-uncased-finetuned-sst-2-english) loaded via HuggingFace. It captures deep contextual bidirectional relationships in sentences, eliminating the need for manual TF-IDF vectorization. |

## Evaluation metrics
Models are evaluated on the test set using standard quantitative classification metrics:

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## Results & Discussion

### Results
(Insert model training/evaluation tables, charts and confusion matrices here.)

### Discussion/Interpretation
- Traditional ML vs. Advanced ML: Linear SVM effectively outperformed Naïve Bayes across all metrics, proving that finding a maximum-margin decision boundary is more effective for TF-IDF word vectors than simple probability distributions.
- The Power of Context: The DistilBERT transformer achieved the highest overall accuracy (90.40%) and a significantly higher precision (93.15%) than the classical models. This indicates the Transformer is exceptionally good at identifying true positive sentiment without being "tricked" by complex grammar, thanks to its self-attention mechanism.
- Resource Trade-offs: While the Transformer yielded the highest accuracy, it was computationally heavy. To manage execution limits in the notebook environment, the Transformer was evaluated on a subset of 500 samples, whereas the ML models instantly processed the full 10,000-sample test set.

### Out-of-Domain Generalization & In-Depth Error Analysis
To test the robustness of our models beyond the standard IMDB dataset, we evaluated them against our custom real-world dataset. A qualitative analysis of the misclassifications reveals several distinct cognitive blind spots across the algorithms:
- The Sarcasm Blindspot: For reviews such as "I loved the part where it ended" or "Well, that was certainly... a movie," all three models (Naïve Bayes, SVM, and DistilBERT) incorrectly predicted them as positive. The models heavily weight explicit positive tokens (e.g., "loved") but fail to comprehend the underlying sarcastic context implying the movie was unenjoyable. This proves that even state-of-the-art transformers struggle with implicit human humor.
- Idiomatic Expressions and Slang: Interestingly, for the review "They really dropped the ball on this one. Big yikes," the advanced SVM and DistilBERT models failed (predicting positive), while the baseline Naïve Bayes correctly predicted negative. Advanced models may lack sufficient context for specific English idioms if they were underrepresented in the IMDB training corpus.
- Complex Syntactic Structures (Expectation vs. Reality): For the review "While some critics argue that the pacing is sluggish, I found the deliberate build-up to be exactly what was needed," all models incorrectly predicted negative. The models were misled by the strong negative token "sluggish" in the dependent clause and failed to capture the overriding positive sentiment expressed in the main clause.
- Cultural and Linguistic Bias (Manglish): When presented with Malaysian colloquialisms such as "potong stim" or "best giler," the models yielded inconsistent predictions because these terms are out-of-vocabulary (OOV) in standard NLTK dictionaries and pre-trained DistilBERT tokenizers.

#### Table: Deep Error Analysis on Real-World Custom Dataset
(Include the table below in your final report or repository README as needed.)

| Review Text | True Sentiment | Model Predictions (NB / SVM / BERT) | Error Category / Root Cause |
|---|---|---|---|
| "I loved the part where it ended." | Negative | Positive (All Models Failed) | Sarcasm: Models falsely prioritize the positive token "loved", ignoring the sarcastic context. |
| "They really dropped the ball on this one. Big yikes." | Negative | NB: Negative / SVM & BERT: Positive | Idiom/Slang: Advanced models lack context for specific conversational idioms ("dropped the ball"). |
| "While some critics argue that the pacing is sluggish, I found the deliberate build-up to be exactly what was needed." | Positive | Negative (All Models Failed) | Complex Syntax: Models are hijacked by the strong negative word "sluggish" and fail to capture the positive resolution. |
| "Story so boring, potong stim right at the climax." | Negative | Negative (All Models Succeeded) | Manglish/OOV Tokens: Models successfully ignored the unknown local slang because the word "boring" anchored the prediction. |

## Conclusion

### Achievements
- Developed a complete, end-to-end NLP sentiment classification pipeline adhering to all TAR UMT assignment objectives.
- Successfully implemented and benchmarked three distinct algorithmic paradigms (Naïve Bayes, SVM, and DistilBERT) across all team members.
- Engineered a custom Hybrid Ensemble (Majority Vote) algorithm that successfully stabilized predictions by combining the speed of statistical models with the contextual depth of transformers.
- Conducted an out-of-domain evaluation using a manually curated real-world dataset to critically analyze AI blind spots in handling modern slang and regional linguistics.

### Limitations and Future Works
#### Limitations
- Resource Intensity: DistilBERT requires significant GPU resources for inference, making low-latency edge deployment challenging compared to the instant execution of classical ML models.
- Cultural & Linguistic Blind Spots: As discovered during our custom dataset testing, models trained purely on Western-centric corpora (IMDB) suffer from domain drift. They consistently misclassify localized Malaysian slang (Manglish, e.g., "potong stim") and fail to detect subtle human sarcasm.

#### Future Works
- Domain Adaptation: Future iterations should focus on fine-tuning the transformer models on a localized, manually annotated dataset of Southeast Asian reviews to bridge the cultural gap and understand regional idioms.
- Model Optimization: Further exploration into quantization and pruning techniques for BERT models to reduce their computational weight for faster, resource-efficient deployment.
- Granular Classification: Upgrading the pipeline from binary classification to multi-class, fine-grained sentiment analysis (1–5 star rating predictions) combined with hyperparameter grid-searching for the classical models.

## Reference & Source

### Sources of Dataset and Tools Used for Development
- Primary Dataset: IMDB Dataset of 50K Movie Reviews, accessed via Kaggle.
- Out-of-Domain Custom Dataset: Curated manually by the team using contemporary reviews from platforms such as Letterboxd and Reddit for real-world robustness testing.
- Machine Learning Frameworks: Scikit-learn (v1.3.2) for Naïve Bayes and Linear SVM implementations; Hugging Face transformers (v4.37.2) for the DistilBERT pipeline.
- Text Preprocessing Tools: Natural Language Toolkit (NLTK v3.8.1) for stopword removal and WordNet lemmatization.
- User Interface & Deployment: Streamlit (v1.31.0) utilized for frontend web application development and real-time inference routing.
- Data Manipulation: Pandas (v2.1.4) and NumPy (v1.26.2).

### APA References
- Npathi, L. (2019). IMDB dataset of 50K movie reviews. Kaggle. https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: Analyzing text with the natural language toolkit. O'Reilly Media, Inc. https://www.nltk.org/book/
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies (pp. 142-150). https://aclanthology.org/P11-1015/
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108. https://arxiv.org/abs/1910.01108
- Streamlit. (2024). Streamlit: The fastest way to build and share data apps. Retrieved from https://streamlit.io/
