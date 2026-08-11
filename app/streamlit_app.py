"""
Drug Review Insight Console — Streamlit deployment prototype.

Pages:
  1) Effectiveness Classifier — benefits review → 3-class (High/Moderate/Low) + sample tip
  2) Side-Effect Topic Matcher — free-text symptoms → LDA topics + sample suggestions

Run from repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Drug Review Insight Console",
    page_icon="DR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
      h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; color: #0b3c4a !important; }
      .stApp {
        background:
          radial-gradient(1200px 500px at 10% -10%, #d7ebe8 0%, transparent 55%),
          radial-gradient(900px 400px at 100% 0%, #f8d9c8 0%, transparent 50%),
          linear-gradient(180deg, #f4f7f6 0%, #eef2f1 100%);
      }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid #c9dbd7;
        border-radius: 12px;
        padding: 0.75rem 1rem;
      }
      .insight-card {
        background: rgba(255,255,255,0.85);
        border-left: 4px solid #2a6f97;
        padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.75rem;
      }
      .warn-card { border-left-color: #ee6c4d; }
      .ok-card { border-left-color: #2a9d8f; }
      .footer-note { color: #4a5c61; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_recommendations() -> dict:
    path = ARTIFACTS / "recommendations.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_topic_labels() -> pd.DataFrame:
    for path in (ARTIFACTS / "model2_topic_labels.csv", PROCESSED / "model2_topic_labels.csv"):
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame(columns=["topic_id", "label", "top_keywords", "n_docs_dominant"])


def try_load_pickle(name: str):
    path = PROCESSED / name
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_effectiveness_model():
    vectorizer = try_load_pickle("tfidf_vectorizer.pkl")
    model = try_load_pickle("model1_log_reg_3.pkl")
    return vectorizer, model


@st.cache_resource
def load_lda_stack():
    count_vectorizer = try_load_pickle("count_vectorizer.pkl")
    lda_path = PROCESSED / "model2_lda.model"
    if count_vectorizer is None or not lda_path.exists():
        return None, None
    from gensim.models import LdaModel

    lda = LdaModel.load(str(lda_path))
    return count_vectorizer, lda


def clean_text(text: str) -> str:
    """Match training prep (lemma + negation-aware stops)."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(
                    f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
                )
            except LookupError:
                nltk.download(pkg, quiet=True)

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english")) - {
            "not",
            "no",
            "nor",
            "n't",
            "never",
            "cannot",
            "can't",
        }
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [
            lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in stop_words and len(tok) > 2
        ]
        return " ".join(tokens)
    except Exception:
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def card(html: str, warn: bool = False, ok: bool = False):
    cls = "insight-card"
    if warn:
        cls += " warn-card"
    elif ok:
        cls += " ok-card"
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


# ---------- Sidebar ----------
recs = load_recommendations()
st.sidebar.title("Insight Console")
st.sidebar.caption("CRISP-DM deployment prototype · 3-class + LDA topics")
page = st.sidebar.radio(
    "Navigate",
    ["Effectiveness Classifier", "Side-Effect Topic Matcher"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<p class="footer-note">{recs["disclaimer"]}</p>',
    unsafe_allow_html=True,
)

# ---------- Page 1 ----------
if page == "Effectiveness Classifier":
    st.title("Effectiveness Classifier")
    st.caption(
        "Paste a benefits-style review. The model predicts High / Moderate / Low "
        "effectiveness (TF-IDF + Logistic Regression) and shows a sample recommendation."
    )

    vectorizer, model = load_effectiveness_model()
    if vectorizer is None or model is None:
        st.error(
            "Model artefacts missing under `data/processed/` "
            "(`tfidf_vectorizer.pkl`, `model1_log_reg_3.pkl`). "
            "Run notebooks `02` and `03` (or the training script) first."
        )
        st.stop()

    examples = {
        "Custom": "",
        "Sounds High": (
            "This medication finally reduced my pain and I could sleep through the night. "
            "Within a week I felt almost normal again and had much more energy."
        ),
        "Sounds Moderate": (
            "I noticed only a moderate improvement. Redness lessened somewhat but symptoms "
            "still come and go and I am not fully better."
        ),
        "Sounds Low": (
            "There was no benefit at all and nothing improved while I was on this drug. "
            "I felt no difference whatsoever."
        ),
    }
    choice = st.selectbox("Example text", list(examples.keys()))
    text = st.text_area(
        "benefitsReview text",
        value=examples[choice],
        height=160,
        placeholder="Describe how the medicine helped (or did not help)…",
    )

    if st.button("Classify review", type="primary"):
        if not text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(text)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            proba = {}
            if hasattr(model, "predict_proba"):
                proba = {
                    str(cls): float(p)
                    for cls, p in zip(model.classes_, model.predict_proba(X)[0])
                }

            tip = recs["effectiveness"].get(pred, {})
            m1, m2 = st.columns(2)
            m1.metric("Predicted class (3-class)", pred)
            if proba:
                conf = max(proba.values())
                m2.metric("Top-class probability", f"{conf:.0%}")

            if pred == "Low":
                card(
                    "<b>Route: human review queue</b> — Low effectiveness predictions "
                    "should be checked by an analyst / clinician pathway before outreach.",
                    warn=True,
                )
            elif pred == "High":
                card(
                    "<b>Route: archive / monitor</b> — High signal; keep for analytics.",
                    ok=True,
                )
            else:
                card(
                    "<b>Route: monitor</b> — Moderate / partial response; follow up if persists."
                )

            if tip:
                card(
                    f"<b>{tip.get('title', pred)}</b><br/>{tip.get('recommendation', '')}"
                )

            if proba:
                st.subheader("Class probabilities")
                st.bar_chart(pd.Series(proba).sort_values(ascending=False))

            with st.expander("Preprocessed text sent to the model"):
                st.code(cleaned or "(empty after cleaning)")

# ---------- Page 2 ----------
else:
    st.title("Side-Effect Topic Matcher")
    st.caption(
        "Describe side effects in free text. The app scores your text against LDA topics "
        "and lists matching themes with sample suggestions."
    )

    count_vectorizer, lda = load_lda_stack()
    topics_df = load_topic_labels()
    threshold = float(recs.get("topic_match_threshold", 0.15))

    if count_vectorizer is None or lda is None:
        st.error(
            "LDA artefacts missing under `data/processed/` "
            "(`count_vectorizer.pkl`, `model2_lda.model`). "
            "Run notebooks `02` and `04` first."
        )
        st.stop()

    if topics_df.empty:
        st.warning("Topic labels CSV not found; showing topic IDs only.")

    label_map = {}
    if not topics_df.empty:
        for _, row in topics_df.iterrows():
            label_map[int(row["topic_id"])] = str(row["label"])
        st.markdown("### Known side-effect topics in this model")
        st.dataframe(
            topics_df[["topic_id", "label", "top_keywords"]],
            use_container_width=True,
            hide_index=True,
        )

    se_examples = {
        "Custom": "",
        "GI / sleep": "After the first dose my stomach felt bad and I had insomnia at night.",
        "Severe systemic": "Severe nausea, headaches, dizziness, fatigue and joint pain.",
        "Skin / acne": "My skin got dry and acne flared up over the past few weeks.",
        "Weight / dryness / sexual": (
            "I gained weight, have dry mouth, and noticed lower libido / sexual side effects."
        ),
    }
    se_choice = st.selectbox("Example symptoms", list(se_examples.keys()))
    se_text = st.text_area(
        "Side effects you suffer",
        value=se_examples[se_choice],
        height=140,
        placeholder="e.g. nausea, can't sleep, dry mouth, weight gain…",
    )
    threshold = st.slider(
        "Topic match threshold",
        min_value=0.05,
        max_value=0.50,
        value=threshold,
        step=0.01,
        help="Only topics with probability ≥ threshold are listed.",
    )

    if st.button("Match side-effect topics", type="primary"):
        if not se_text.strip():
            st.warning("Please describe your side effects first.")
        else:
            cleaned = clean_text(se_text)
            X = count_vectorizer.transform([cleaned])
            if X.sum() == 0:
                st.warning(
                    "After cleaning, no vocabulary terms matched the model. "
                    "Try more specific symptom words (nausea, headache, rash, insomnia…)."
                )
            else:
                # Build gensim bow using CountVectorizer column indices
                X_csr = X.tocsr()
                idxs = X_csr.indices
                vals = X_csr.data
                bow = [(int(j), float(v)) for j, v in zip(idxs, vals)]
                dist = lda.get_document_topics(bow, minimum_probability=0.0)
                dist = sorted(dist, key=lambda x: x[1], reverse=True)

                matched = [(tid, float(p)) for tid, p in dist if float(p) >= threshold]
                st.subheader("Matched side-effect topics")
                if not matched:
                    st.info(
                        f"No topics scored ≥ {threshold:.0%}. "
                        "Try lowering the threshold or adding more symptom detail."
                    )
                    # still show top-3 for transparency
                    matched = [(tid, float(p)) for tid, p in dist[:3]]
                    st.caption("Showing top 3 topics below threshold for reference:")

                rows = []
                for tid, p in matched:
                    label = label_map.get(tid, f"Topic {tid}")
                    tip_block = recs["side_effect_topics"].get(str(tid), {})
                    suggestion = tip_block.get(
                        "suggestion",
                        "Sample tip: discuss these symptoms with a clinician or pharmacist.",
                    )
                    rows.append(
                        {
                            "topic_id": tid,
                            "label": tip_block.get("label", label),
                            "probability": p,
                            "suggestion": suggestion,
                        }
                    )
                    warn = tid == 2  # severe systemic topic
                    card(
                        f"<b>Topic {tid}: {tip_block.get('label', label)}</b> "
                        f"(score {p:.0%})<br/>{suggestion}",
                        warn=warn,
                        ok=(tid == 1),
                    )

                chart_df = pd.DataFrame(rows).set_index("label")["probability"]
                st.bar_chart(chart_df)

                with st.expander("Preprocessed text sent to LDA"):
                    st.code(cleaned or "(empty after cleaning)")
                with st.expander("Full topic distribution"):
                    full = pd.DataFrame(
                        [
                            {
                                "topic_id": tid,
                                "label": label_map.get(tid, f"Topic {tid}"),
                                "probability": float(p),
                            }
                            for tid, p in dist
                        ]
                    ).sort_values("probability", ascending=False)
                    st.dataframe(full, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "XBDS2024N CRISP-DM deployment · "
    "`streamlit run app/streamlit_app.py` · not medical advice"
)
