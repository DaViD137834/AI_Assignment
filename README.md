# BMCS2074 Artificial Intelligence — Group Assignment
## Comparative Sentiment Analysis on IMDB Movie Reviews: From Classical Machine Learning to Transformer Architectures

**Tutorial Group:** G5
**Tutor:** Dr Cheng Kam Ching
**Session:** 202605, Year 2026/27
**Programme:** RDS Y2S1 (Bachelor of Data Science)

### Team Members

| No | Student Name | Student ID |
|----|---------------|------------|
| 1 | Ambrose Teo Chen Bin | 2612472 |
| 2 | David Lee Yong Fu | 2612486 |
| 3 | Teh Zi Yan | 2612547 |

### Purpose

This project benchmarks three sentiment classification paradigms on the IMDB Large Movie Review
Dataset — a probabilistic baseline (Naïve Bayes), a classical margin-based classifier (Linear SVM),
and a pre-trained transformer (DistilBERT) — to evaluate the trade-off between accuracy,
precision/recall balance, and computational cost. The models are also stress-tested on a manually
curated out-of-domain dataset (Gen-Z slang, sarcasm, Manglish) to surface real-world blind spots, and
combined into a majority-vote hybrid ensemble.

---

## 1. Main Prototype Functions

- **Text preprocessing pipeline**: HTML/URL stripping, lowercasing, stopword removal, lemmatization
  (NLTK), sequence padding/truncation.
- **Model training & inference**:
  - Naïve Bayes (MultinomialNB) on TF-IDF features (max 10,000 features)
  - Linear SVM (LinearSVC) on TF-IDF features
  - DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) via HuggingFace pipeline
- **Hybrid ensemble**: majority-vote combination of the three models.
- **Evaluation module**: Accuracy, Precision, Recall, F1-Score on the IMDB test set.
- **Out-of-domain error analysis**: qualitative testing against a 25-review custom dataset
  (sarcasm, idioms, complex syntax, Manglish/OOV tokens).
- **Streamlit front-end**: web UI for real-time sentiment inference. *[confirm entry-point filename, e.g. `app.py`]*

---

## 2. Tech Stack & Tool Versions

| Component | Tool | Version |
|---|---|---|
| Language | Python | *[fill in, e.g. 3.10]* |
| ML framework | scikit-learn | 1.3.2 |
| Transformers | Hugging Face `transformers` | 4.37.2 |
| NLP preprocessing | NLTK | 3.8.1 |
| Web UI | Streamlit | 1.31.0 |
| Data handling | pandas | 2.1.4 |
| Data handling | NumPy | 1.26.2 |

---

## 3. Supported Operating System / Execution Environment

- Tested on: *[e.g. Windows 11 / macOS / Ubuntu 22.04 — fill in]*
- Also runnable on: Jupyter Notebook / Google Colab (GPU runtime recommended for the DistilBERT
  section)

---

## 4. System Requirements

- **Runtime:** Python *[version]*
- **Recommended tools:** Anaconda or venv, VS Code or Jupyter Notebook / Google Colab
- **Internet access:** Required on first run to download the DistilBERT model weights from
  HuggingFace Hub and (if not bundled) the IMDB dataset from Kaggle.
- **Hardware:**
  - CPU: sufficient for Naïve Bayes and SVM (near-instant on the full 10,000-sample test set)
  - GPU: recommended for DistilBERT inference — the notebook environment evaluated it on a
    subset of 500 samples due to execution-time limits without one
  - RAM: *[fill in, e.g. 8GB+]*
- **Approximate timings** *(fill in with your actual measured times)*:
  - Environment/dependency installation: ~__ min
  - Dataset load + preprocessing: ~__ min
  - Naïve Bayes / SVM training: ~__ sec–min
  - DistilBERT inference (500 samples): ~__ min

---

## 5. Installation

1. Clone or extract the project folder.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(See `requirements.txt` below — update with your actual pinned versions.)*
4. Download required NLTK resources (one-time):
   ```bash
   python -m nltk.downloader stopwords wordnet punkt
   ```

### requirements.txt (suggested contents — verify against your actual imports)
```
scikit-learn==1.3.2
transformers==4.37.2
nltk==3.8.1
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.2
torch          # required by transformers/DistilBERT — pin the version you used
```

---

## 6. Dataset & Trained-Model Setup

- **Primary dataset:** IMDB Dataset of 50K Movie Reviews (Maas et al.), accessed via Kaggle.
  - Download link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - Place the CSV at: *[fill in expected path, e.g. `data/IMDB_Dataset.csv`]*
- **Custom out-of-domain dataset:** 25 manually curated contemporary reviews (Letterboxd/Reddit
  sourced), included at: *[fill in path, e.g. `data/custom_reviews.csv`]*
- **Trained models:**
  - Naïve Bayes / SVM: trained from scratch on script run (fast, no pre-saved weights needed) —
    or specify if you saved `.pkl` files and where.
  - DistilBERT: pre-trained weights (`distilbert-base-uncased-finetuned-sst-2-english`) auto-downloaded
    from HuggingFace Hub on first run (requires internet).

---

## 7. Running the Prototype

*(Fill in with your actual filenames/commands — placeholders below)*

```bash
# Run the full training + evaluation pipeline
python main.py

# OR launch the Streamlit web app
streamlit run app.py
```

---

## 8. Test-Input Instructions & Expected Output

- **Input:** a single movie review string (via Streamlit text box) or a CSV of reviews (batch mode).
- **Example input:** `"While some critics argue that the pacing is sluggish, I found the deliberate build-up to be exactly what was needed."`
- **Expected output:** predicted sentiment label (`Positive` / `Negative`) from each of the three
  models plus the ensemble majority vote, and confidence/probability scores where available.
- **Evaluation mode output:** Accuracy, Precision, Recall, F1-Score printed/displayed per model
  (see Section 4 Results in the report for reference numbers: NB 0.855 / SVM 0.883 / DistilBERT 0.904
  accuracy).

---

## 9. Known Limitations & Troubleshooting

- **Sarcasm blind spot:** all three models misclassify sarcastic reviews (e.g. "I loved the part
  where it ended") as positive.
- **Idiom/slang gaps:** advanced models (SVM, DistilBERT) can fail on idioms like "dropped the ball"
  that are underrepresented in the IMDB training corpus.
- **Manglish/OOV tokens:** Malaysian colloquialisms (e.g. "potong stim", "best giler") are
  out-of-vocabulary for NLTK and the pre-trained DistilBERT tokenizer, producing inconsistent
  predictions.
- **Resource intensity:** DistilBERT requires notably more compute; without a GPU, inference was
  limited to a 500-sample subset instead of the full 10,000-sample test set.
- **Common issues:**
  - `ModuleNotFoundError` → re-run `pip install -r requirements.txt`.
  - NLTK `LookupError` (missing corpus) → re-run the NLTK downloader command in Section 5.
  - Slow/hanging DistilBERT step → ensure a GPU runtime is available (e.g. Colab GPU), or reduce
    the evaluation subset size.
  - No internet access → DistilBERT weights cannot be downloaded on first run; classical models
    (NB/SVM) can still run offline once the dataset is local.

---

## 10. Member-to-File Mapping

*(Required by the submission guide — please complete this table with your actual repo structure)*

| Member | Files/Modules Owned | Component |
|---|---|---|
| David Lee Yong Fu | *[e.g. `distilbert_model.ipynb`]* | DistilBERT transformer pipeline |
| Ambrose Teo Chen Bin | *[fill in]* | *[fill in]* |
| Teh Zi Yan | *[fill in]* | *[fill in]* |

---

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word
  vectors for sentiment analysis. *ACL-HLT 2011*. https://aclanthology.org/P11-1015/
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT.
  *arXiv:1910.01108*. https://arxiv.org/abs/1910.01108
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly.
- Streamlit. (2024). https://streamlit.io/
