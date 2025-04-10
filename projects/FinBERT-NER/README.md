# 🏷️ Financial Named Entity Recognition with FinBERT

Manual tagging of financial entities in earnings reports and 10-K filings is labor-intensive and error-prone. This project seeks to automate the identification of domain-specific financial entities using FinBERT, a transformer model pretrained on financial corpora.
This project fine-tunes [FinBERT](https://huggingface.co/ProsusAI/finbert), a domain-specific transformer model, to perform token-level Named Entity Recognition (NER) on financial disclosures. Using the [Financial-NER-NLP](https://huggingface.co/datasets/Josephgflowers/Financial-NER-NLP) dataset, the model learns to extract structured financial entities such as interest rates, borrowing capacity, and equity instruments from unstructured financial statements.

---

## 📊 Project Highlights

- **Model:** FinBERT (`ProsusAI/finbert`) fine-tuned for token classification
- **Dataset:** Josephgflowers/Financial-NER-NLP (XBRL-annotated financial text)
- **Task:** Named Entity Recognition (NER) using IOB2 tagging
- **Results:**  
  - Baseline F1-score: **48%**  
  - Final F1-score: **58.41%**  
  - Precision: **56.17%**, Recall: **60.82%**
- **Tech Stack:** HuggingFace Transformers, Datasets, PyTorch, Scikit-learn, NLTK, Google Colab
