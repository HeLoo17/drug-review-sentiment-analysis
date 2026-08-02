"""
Drug Review Insight Console — Streamlit deployment prototype.

Run from repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PROCESSED = ROOT / "data" / "processed"
VISUALS = ROOT / "visuals"

st.set_page_config(
    page_title="Drug Review Insight Console",
    page_icon="DR",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual direction: deep teal + coral (matches project charts), soft paper background
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap');
      html, body, [class*="css"]  {
        font-family: 'Source Sans 3', sans-serif;
      }
      h1, h2, h3 {
        font-family: 'Fraunces', Georgia, serif !important;
        color: #0b3c4a !important;
      }
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
        background: rgba(255,255,255,0.8);
        border-left: 4px solid #2a6f97;
        padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.75rem;
      }
      .warn-card {
        border-left-color: #ee6c4d;
      }
      .footer-note {
        color: #4a5c61;
        font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_comparison() -> pd.DataFrame:
    path = ARTIFACTS / "model1_3vs5_comparison.csv"
    if not path.exists():
        path = PROCESSED / "model1_3vs5_comparison.csv"
    return pd.read_csv(path)


@st.cache_data
def load_topics() -> pd.DataFrame:
    path = ARTIFACTS / "model2_topic_labels.csv"
    if not path.exists():
        path = PROCESSED / "model2_topic_labels.csv"
    return pd.read_csv(path)


def try_load_pickle(name: str):
    path = PROCESSED / name
    if not path.exists():
        return None
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scoring_artifacts():
    vectorizer = try_load_pickle("tfidf_vectorizer.pkl")
    model = try_load_pickle("model1_log_reg_3.pkl")
    return vectorizer, model


def clean_text(text: str) -> str:
    """Match training prep as closely as possible (lemma + negation-aware stops)."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
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


def show_image(name: str, caption: str | None = None):
    path = VISUALS / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: `visuals/{name}` — run the notebooks to regenerate.")


# ---------- Sidebar ----------
st.sidebar.title("Insight Console")
st.sidebar.caption("CRISP-DM deployment prototype · Druglib.com reviews")
page = st.sidebar.radio(
    "Navigate",
    [
        "Research Overview",
        "Effectiveness Insights",
        "Side-Effect Topics",
        "Live Review Scoring",
        "Deployment Architecture",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p class="footer-note">Research prototype only — not medical advice. '
    "Predictions support triage research, not clinical decisions.</p>",
    unsafe_allow_html=True,
)

# ---------- Pages ----------
if page == "Research Overview":
    st.title("Drug Review Insight Console")
    st.subheader("What this research found")

    st.markdown(
        """
        Patient free-text reviews carry signals that star ratings alone miss.
        This project applies **CRISP-DM** to Druglib.com reviews with two complementary models:
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="insight-card"><b>Model 1 — Effectiveness classification</b><br/>'
            "Predict perceived effectiveness from <code>benefitsReview</code> "
            "(TF-IDF + Logistic Regression / Linear SVM). "
            "Compared <b>5-class</b> vs binned <b>3-class</b> (High / Moderate / Low).</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="insight-card"><b>Model 2 — Side-effect topic modeling</b><br/>'
            "Discover latent themes in <code>sideEffectsReview</code> with <b>LDA</b> "
            "(k=5–8 coherence sweep). Topics are manually labeled from top keywords.</div>",
            unsafe_allow_html=True,
        )

    try:
        comparison = load_comparison()
        topics = load_topics()
        lr3 = comparison[
            (comparison["scheme"] == "3-class") & (comparison["model"] == "LR (3-class)")
        ].iloc[0]
        lr5 = comparison[
            (comparison["scheme"] == "5-class") & (comparison["model"] == "LR (5-class)")
        ].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("3-class LR macro-F1", f"{lr3['macro_f1']:.2f}")
        m2.metric("5-class LR macro-F1", f"{lr5['macro_f1']:.2f}")
        m3.metric("LDA topics (best k)", f"{len(topics)}")
        m4.metric("Reviews tagged (train)", f"{int(topics['n_docs_dominant'].sum()):,}")
    except Exception as e:
        st.warning(f"Could not load summary metrics: {e}")

    st.markdown("### Key takeaways")
    st.markdown(
        """
        1. **3-class labels outperform 5-class on macro-F1** — adjacent effectiveness tiers are hard to separate from short benefits text.
        2. Both LR and SVM **beat a majority-class baseline** on macro-F1 under class imbalance.
        3. Side-effect text clusters into interpretable themes (GI/onset, severe systemic, skin/acne, weight/sexual/dryness, minimal language).
        4. Best deployment fit: **3-class LR for triage** + **LDA topic tags** for theme filters, with human review for Low / severe topics.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        show_image("evaluation_model1_metrics.png", "Model 1 — accuracy & macro-F1 by scheme")
    with col_b:
        show_image("evaluation_model2_topics.png", "Model 2 — dominant topic volumes")

elif page == "Effectiveness Insights":
    st.title("Model 1 — Effectiveness classification")
    st.caption("Input: benefitsReview (TF-IDF) · Targets: 5-class vs 3-class")

    comparison = load_comparison()
    st.dataframe(
        comparison.style.format(
            {
                "accuracy": "{:.3f}",
                "macro_precision": "{:.3f}",
                "macro_recall": "{:.3f}",
                "macro_f1": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        '<div class="insight-card"><b>Insight:</b> Macro-F1 rises from ~0.43 (5-class LR) to ~0.56 (3-class LR). '
        "Accuracy looks high for the 3-class majority baseline (~0.70) because High dominates — "
        "macro-F1 is the fairer comparison and shows real model lift.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        show_image("model1_3vs5_comparison.png", "Macro-F1 comparison chart")
    with c2:
        show_image("model1_confusion_matrices.png", "Confusion matrices (LR/SVM × 3/5-class)")

    st.markdown("### Label schemes")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **5-class (original)**  
            Highly · Considerably · Moderately · Marginally · Ineffective
            """
        )
    with right:
        st.markdown(
            """
            **3-class (binned)**  
            - **High** ← Highly + Considerably  
            - **Moderate** ← Moderately  
            - **Low** ← Marginally + Ineffective
            """
        )

elif page == "Side-Effect Topics":
    st.title("Model 2 — Side-effect topics (LDA)")
    st.caption("Input: sideEffectsReview · Technique: LDA with c_v coherence sweep (k=5…8)")

    topics = load_topics()
    st.dataframe(topics, use_container_width=True)

    st.bar_chart(
        topics.set_index("label")["n_docs_dominant"].sort_values(ascending=True),
        horizontal=True,
    )

    st.markdown(
        '<div class="insight-card warn-card"><b>Insight:</b> The largest topic is often '
        "<i>minimal / generic side-effect language</i> — useful as a filter, but severe themes "
        "(pain/nausea/fatigue; weight/sexual/dryness) matter more for pharmacovigilance triage.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        show_image("model2_coherence_sweep.png", "Coherence sweep — best k selected")
    with c2:
        show_image("model2_topic_keywords.png", "Top keywords per topic")
    show_image("model2_topic_distribution.png", "Dominant-topic distribution")

elif page == "Live Review Scoring":
    st.title("Live review scoring")
    st.caption("Score a benefits review with the trained 3-class Logistic Regression model.")

    vectorizer, model = load_scoring_artifacts()
    if vectorizer is None or model is None:
        st.warning(
            "Model artifacts not found under `data/processed/`. "
            "Run notebooks `02` and `03` first, then restart this app to enable live scoring. "
            "Research charts on other pages still work from committed visuals/artifacts."
        )
    else:
        examples = {
            "Positive benefits": "This medication finally reduced my pain and I could sleep through the night.",
            "Limited benefit": "I noticed only a slight improvement and not much change in my symptoms.",
            "No benefit": "There was no benefit at all and nothing improved while I was on this drug.",
            "Custom": "",
        }
        choice = st.selectbox("Example text", list(examples.keys()))
        default = examples[choice]
        text = st.text_area("benefitsReview text", value=default, height=140)

        if st.button("Score review", type="primary") and text.strip():
            cleaned = clean_text(text)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            proba = {}
            if hasattr(model, "predict_proba"):
                proba = {
                    cls: float(p)
                    for cls, p in zip(model.classes_, model.predict_proba(X)[0])
                }

            route = "human_review" if pred == "Low" else "archive"
            st.markdown(f"### Predicted effectiveness: **{pred}**")
            if proba:
                st.write("Class probabilities")
                st.bar_chart(pd.Series(proba).sort_values(ascending=False))

            if route == "human_review":
                st.markdown(
                    '<div class="insight-card warn-card"><b>Route: human review queue</b> — '
                    "Low effectiveness predictions should be checked by an analyst before any outreach.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="insight-card"><b>Route: searchable archive</b> — '
                    "store prediction + optional LDA topic tags for dashboards.</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("Preprocessed text sent to the model"):
                st.code(cleaned or "(empty after cleaning)")

        st.markdown("---")
        st.markdown("### Suggested dual-model response payload")
        st.json(
            {
                "review_id": "demo-001",
                "effectiveness_3class": "High | Moderate | Low",
                "side_effect_topic": "attach LDA dominant topic in full service",
                "route": "human_review if Low or severe topic else archive",
            }
        )

elif page == "Deployment Architecture":
    st.title("Suggested deployment architecture")
    st.markdown(
        """
        **Drug Review Insight Console** — batch or API scoring for pharmacovigilance /
        patient-support teams, with human-in-the-loop review for Low effectiveness and
        high-concern side-effect topics.
        """
    )
    show_image("deployment_architecture.png", "Conceptual architecture")

    st.markdown("### Workflow")
    st.markdown(
        """
        1. Ingest review text (`benefitsReview`, `sideEffectsReview`)
        2. Clean + vectorize with training-time artifacts
        3. Score Model 1 (3-class triage; optional 5-class detail)
        4. Infer Model 2 topic mixture / dominant label
        5. Route Low / severe-topic items to analysts
        6. Monitor drift and retrain with feedback labels
        """
    )

    st.markdown(
        '<div class="insight-card warn-card"><b>Production caveats:</b> UCI Druglib data is '
        "non-commercial research use; outputs are not medical advice; monitor class priors and "
        "topic drift; treat free text as sensitive.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("CRISP-DM notebooks 00–06 · streamlit run app/streamlit_app.py")
