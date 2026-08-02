# Drug Review Sentiment Analysis

Text mining project (CRISP-DM) on patient drug reviews — effectiveness classification + side-effect topic modeling.

## Dataset
[Drug Review Dataset (Druglib.com)](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)

UCI Machine Learning Repository. Not included in this repo due to license restrictions (non-commercial research use only). Download the train/test TSV files and place them in `data/` before running:

- `data/drugLibTrain_raw.tsv`
- `data/drugLibTest_raw.tsv`

## Project Structure

```
notebooks/
  00_business_understanding.ipynb   # Business problem, stakeholders, ethics
  01_data_understanding.ipynb       # EDA, distributions, word clouds
  02_data_preparing.ipynb           # Cleaning, TF-IDF, DTM, 3-class labels
  03_model_1_modeling.ipynb         # LR / SVM — 5-class vs 3-class comparison
  04_model_2_topic_modeling.ipynb   # LDA side-effect topics (k=5–8)
  05_evaluation.ipynb               # Cross-model evaluation
  06_deployment.ipynb               # Suggested Insight Console architecture
visuals/                            # Figures exported by notebooks
data/                               # Raw + processed artifacts (gitignored)
requirements.txt
```

## Methodology
This project follows CRISP-DM:

1. **Business Understanding** — effectiveness prediction + side-effect theme discovery for triage
2. **Data Understanding** — ~3.1k train / ~1.0k test reviews; imbalanced effectiveness labels
3. **Text Data Preparation** — clean/lemmatize; TF-IDF (Model 1); CountVectorizer DTM (Model 2); 3-class binning
4. **Modeling** — Model 1: Logistic Regression & Linear SVM (3- vs 5-class); Model 2: LDA topics
5. **Evaluation** — accuracy / macro-F1 vs majority baseline; topic coherence + manual labels
6. **Deployment** — conceptual Drug Review Insight Console (batch/API + human review)

## How to Run

```bash
python -m pip install -r requirements.txt
# Download Druglib TSVs into data/
cd notebooks
jupyter notebook   # run 00 → 06 in order
```

Processed matrices and models are written to `data/processed/` (ignored by git). Figures are saved under `visuals/`.

## Results

**Model 1 — effectiveness from `benefitsReview` (held-out test)**

| Scheme | Model | Accuracy | Macro-F1 |
|--------|-------|----------|----------|
| 5-class | Majority baseline | 0.40 | 0.11 |
| 5-class | Logistic Regression | 0.47 | 0.43 |
| 5-class | Linear SVM | 0.46 | 0.40 |
| 3-class | Majority baseline | 0.70 | 0.27 |
| 3-class | Logistic Regression | 0.68 | **0.56** |
| 3-class | Linear SVM | 0.71 | 0.55 |

3-class labels (High / Moderate / Low) improve macro-F1 versus 5-class by reducing adjacent-class confusion. Prefer 3-class LR for triage; keep 5-class when fine granularity is required.

**Model 2 — LDA on `sideEffectsReview`**

- Topic sweep k=5..8; best **k=5** (c_v coherence ≈ 0.73)
- Themes include: GI/onset & sleep disruption; minimal/generic language; severe pain/nausea/fatigue; skin/acne; weight/dryness/sexual effects

## License
Code is licensed under MIT. Dataset usage is subject to UCI's terms — see link above.
