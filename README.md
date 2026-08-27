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
| Language | Python | 3.12.7 |
| ML framework | scikit-learn | 1.3.2 |
| Transformers | Hugging Face `transformers` | 4.37.2 |
| NLP preprocessing | NLTK | 3.8.1 |
| Web UI | Streamlit | 1.31.0 |
| Data handling | pandas | 2.1.4 |
| Data handling | NumPy | 1.26.2 |

---

## 3. Supported Operating System / Execution Environment

- Tested on: *Windows 11 Home / macOS Sonoma / Ubuntu 22.04 LTS*
- Also runnable on: Jupyter Notebook / Google Colab (GPU runtime recommended for the DistilBERT
  section)

---

## 4. System Requirements

- **Runtime:** Python 3.12+(compatible with Python 3.9–3.12)
- **Recommended tools:** Anaconda or venv, VS Code or Jupyter Notebook / Google Colab
- **Internet access:** Required on first run to download the DistilBERT model weights from
  HuggingFace Hub and (if not bundled) the IMDB dataset from Kaggle.
- **Hardware:**
  - CPU: Multi-core processor (sufficient for Naïve Bayes, Linear SVM, and single-review inference)
  - GPU: Recommended for DistilBERT batch evaluation (the notebook evaluated DistilBERT on a
    500-sample subset due to CPU execution-time constraints)
  - RAM: 8GB minimum (16GB recommended for smooth Transformer loading)
- **Approximate timings** *(fill in with your actual measured times)*:
  - Environment/dependency installation: ~2-3 min
  - Dataset load + preprocessing: ~15–20 sec
  - Naïve Bayes / SVM training: ~5–10 sec
  - DistilBERT inference (500 samples): ~1–2 min (GPU) / ~4–5 min (CPU)

---

## 5. Installation

1. Extract the project ZIP folder RDS2S1G5_G3_ComparativeSentimentAnalysis.zip.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r 05_Installation_and_User_Guide/requirements.txt
   ```
   *(See `requirements.txt` below — update with your actual pinned versions.)*
4. Download required NLTK resources (one-time):
   ```bash
   python -m nltk.downloader stopwords wordnet
   ```

### requirements.txt (suggested contents — verify against your actual imports)
```
sstreamlit==1.31.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
nltk==3.8.1
transformers==4.37.2
torch==2.2.0
joblib==1.3.2
```

---

## 6. Dataset & Trained-Model Setup

- **Primary dataset:** IMDB Dataset of 50K Movie Reviews (Maas et al.), accessed via Kaggle.
  - Download link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - Place the CSV at: `03_Dataset/IMDBDataset.csv`
- **Custom out-of-domain dataset:** 25 manually curated contemporary reviews (Letterboxd/Reddit
  sourced), included at: `03_Dataset/real_world_test.csv`
- **Trained models:**
  - Naïve Bayes / Linear SVM: Vectorizer and model binaries pre-saved at `04_Trained_Model/vectorizer.pkl`, `04_Trained_Model/nb_model.pkl`, and       `04_Trained_Model/svm_model.pkl`.
  - DistilBERT: pre-trained weights (`distilbert-base-uncased-finetuned-sst-2-english`) auto-downloaded
    from HuggingFace Hub on first run (requires internet).

---

## 7. Running the Prototype

*(Fill in with your actual filenames/commands — placeholders below)*

```bash
# Option 1: Run full model training & evaluation notebook
jupyter notebook 02_Source_Code/AI_Assignment.ipynb

# Option 2: Launch the interactive Streamlit web application
streamlit run 02_Source_Code/app.py
```

---

## 8. Test-Input Instructions & Expected Output

- **Input:** Single movie review string (via Streamlit text area) or batch evaluation in notebook.
- **Example input:** `"While some critics argue that the pacing is sluggish, I found the deliberate build-up to be exactly what was needed."`
- **Expected output:** predicted sentiment label (`Positive` / `Negative`) from each of the three
  models plus the ensemble majority vote, and confidence/probability scores where available.
- ## Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Naïve Bayes** | 0.8547 | 0.8556 | 0.8561 | 0.8559 |
| **Linear SVM** | 0.8834 | 0.8801 | **0.8899** | 0.8849 |
| **DistilBERT** | **0.9040** | **0.9315** | 0.8608 | **0.8947** |

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
| Ambrose Teo Chen Bin | `02_Source_Code/AI_Assignment.ipynb`<br>`03_Dataset/cleaned_dataset.csv`<br>`03_Dataset/IMDBDataset.csv` | Text preprocessing pipeline (Regex/NLTK), Linear SVM classifier implementation, and notebook training workflow. |
| David Lee Yong Fu | `02_Source_Code/AI_Assignment.ipynb`<br>`03_Dataset/cleaned_dataset.csv`<br>`03_Dataset/real_world_test.csv` | Naïve Bayes baseline model implementation, out-of-domain dataset collection, and error analysis. |
| Teh Zi Yan | `02_Source_Code/AI_Assignment.ipynb`<br>`03_Dataset/cleaned_dataset.csv`<br>`02_Source_Code/app.py` | DistilBERT transformer integration, pipeline evaluation module, and Hybrid Ensemble (Majority Vote) logic implementation. |

