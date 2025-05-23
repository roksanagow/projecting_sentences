# Sense-Annotated Sentence Projection and WiC Formatting Toolkit

This project provides tools to create sense-annotated sentences for polysemous words in low-resource languages and convert them into the Word-in-Context (WiC) format for model training. It supports interactive sentence annotation via embedding visualization, and automatic generation of balanced WiC-style sentence pairs.

---

## 🔍 Project Overview

The goal of this project is to:

- Assist in building sense-annotated datasets from raw example sentences.
- Visualize and annotate these examples interactively.
- Convert the annotated data into binary sentence pairs following the WiC format, suitable for training models on contextual word sense disambiguation.

---

## 📂 Project Structure

```
.
├── formatted_wic_files_16_reps/         # Output WiC-formatted files (example)
├── formatted_wsd_files/                 # Saved annotated data (example)
├── functions.py                         # Utility functions used by notebooks
├── projecting_sentences_for_annotation.ipynb  # Interactive annotation interface
├── wic_formatting.ipynb                 # Conversion of annotations to WiC format
├── requirements.txt                     # Required Python packages
└── README.md                            # Project documentation
```

---

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start with Annotation**:
   Open the notebook: `projecting_sentences_for_annotation.ipynb`

---

## 🧠 Step-by-Step Guide

### 1. 📌 Sentence Annotation

**Notebook**: `projecting_sentences_for_annotation.ipynb`

This notebook enables interactive sense annotation of sentences using 2D projections of contextual embeddings.

#### How it works:
- Load your list of example sentences containing a polysemous word (e.g., "bank").
- Embed the sentences using a multilingual language model (e.g., mBERT).
- Project the sentence embeddings using a dimensionality reducer:
  - `UMAP`
  - `t-SNE`
  - `MDS`
  - `PCA`
- Visualize the sentences in 2D space:
  - Hover to view the sentence.
  - Click to annotate (up to 5 senses: labeled `0–4`).
  - Change labels with **"Next Label"**.
  - Use **"Finish Labelling"** to view annotation counts.
  - Save as CSV with: `saved_sentences/{word}_labelled_sentences.csv`

**CSV format**:
```
lemma,sentence,sense,start,end
```

---

### 2. 🔁 WiC Data Formatting

**Notebook**: `wic_formatting.ipynb`

Converts your saved, sense-annotated sentences into the **WiC** format for binary classification model training.

#### Input:
A CSV with the following columns:
```
lemma,sentence,sense,start,end
```

#### Key features:
- Limits sentence repetition per word (customizable).
- Accepts a random seed for reproducibility.
- Ensures sentence-level train/val/test splits (no overlap).
- Balances positive and negative WiC pairs per word.

#### Algorithm Summary:

1. **Word Splitting**  
   - 70% train, 15% dev, 15% test

2. **Sentence Redistribution**  
   - 30% of training words appear in all sets  
   - Ensures no sentence overlap and preserves sense balance

3. **WiC Pairing**  
   - Pair sentences up to 16 times  
   - Balanced mix of same-sense and different-sense pairs

This procedure ensures robust and balanced training/testing data, even for low-resource settings.

---

## 📊 Example Output

Each pair in the final dataset contains:
- Two sentences with the same target word
- Their respective indices and start-end spans
- A binary label indicating same/different senses

Example:
```
lemma, sentence1, start1, end1, sentence2, start2, end2, label
...
```

---

## 🧩 Dependencies

All required Python packages are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

---

## 📘 Citation

If you use this toolkit in your work, please cite the original paper [TODO].

---

## 🧑‍💻 Contact

For questions, feedback, or collaboration inquiries, feel free to contact me.

