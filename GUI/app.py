# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="URL Phishing Detection",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 URL-Based Phishing Detection System")
st.markdown("**Detect malicious URLs using machine learning**")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR = ROOT_DIR / "Main_Model"

def _pick(*cands):
    for p in cands:
        if p and p.is_file():
            return p
    return None

# -----------------------------------------------------------------------------
# Model + Scaler loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    p = _pick(MODEL_DIR/"model.pkl", APP_DIR/"model.pkl", ROOT_DIR/"model.pkl")
    if not p:
        st.error("model.pkl not found")
        st.caption(f"Tried: {[str(x) for x in [MODEL_DIR/'model.pkl', APP_DIR/'model.pkl', ROOT_DIR/'model.pkl']]}")
        return None
    return joblib.load(p)

@st.cache_resource
def load_scaler():
    p = _pick(MODEL_DIR/"scaler.pkl", APP_DIR/"scaler.pkl", ROOT_DIR/"scaler.pkl")
    if not p:
        # Only needed when the model has no internal scaler
        st.caption("External scaler.pkl not found. If your model embeds its own scaler, this is fine.")
        return None
    return joblib.load(p)

model = load_model()
scaler = load_scaler()

def _uses_internal_scaler(m):
    # Your UltimateOptimizedModel exposes both .scaler and .stacking_model
    return hasattr(m, "scaler") and hasattr(m, "stacking_model")

# -----------------------------------------------------------------------------
# Canonical training feature names
# -----------------------------------------------------------------------------
def _resolve_expected(model, scaler):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # fallback: your manual list of 30 features
    return [
        "url_length","domain_age","subdomain_count","special_chars",
        "https_usage","google_index","page_rank","domain_registration_length",
        "suspicious_keywords","dots_count","hyphens_count","underscores_count",
        "slashes_count","question_marks","equal_signs","at_symbols",
        "ampersands","percent_signs","hash_signs","digits_count",
        "letters_count","alexa_rank","domain_trust","ssl_certificate",
        "redirects_count","page_load_time","has_forms","hidden_elements",
        "external_links_ratio","image_text_ratio"
    ]

EXPECTED = _resolve_expected(model, scaler)

# OPTIONAL diagnostics
st.caption(f"len(EXPECTED)={len(EXPECTED)}  EXPECTED={EXPECTED}")
try:
    base = getattr(model, "stacking_model", None)
    if base and hasattr(base, "estimators_"):
        ns = {name: getattr(est, "n_features_in_", None) for name, est in base.named_estimators_.items()}
        st.caption(f"Base estimator feature counts: {ns}")
except Exception:
    pass
if hasattr(scaler, "feature_names_in_"):
    st.caption(f"External scaler expects: {list(scaler.feature_names_in_)}")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Feature Input", "URL Analysis", "Batch Prediction"]
)

# -----------------------------------------------------------------------------
# Manual Feature Input
# -----------------------------------------------------------------------------
if input_method == "Manual Feature Input":
    st.header("Manual Feature Input")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("URL Structure Features")
        url_length = st.number_input("URL Length", min_value=0, max_value=1000, value=50)
        domain_age = st.number_input("Domain Age (days)", min_value=0, max_value=10000, value=365)
        subdomain_count = st.number_input("Subdomain Count", min_value=0, max_value=10, value=1)
        special_chars = st.number_input("Special Character Count", min_value=0, max_value=100, value=5)
    with col2:
        st.subheader("Security Features")
        https_usage = st.selectbox("HTTPS Usage", [0, 1], format_func=lambda x: "Yes" if x else "No")
        google_index = st.selectbox("Google Indexed", [0, 1], format_func=lambda x: "Yes" if x else "No")
        page_rank = st.slider("Page Rank", 0.0, 10.0, 5.0)
        domain_registration_length = st.number_input("Domain Registration Length (years)", min_value=1, max_value=100, value=1)

    st.subheader("Additional URL Features")
    col3, col4, col5 = st.columns(3)
    with col3:
        feat9  = st.number_input("Suspicious Keywords Count", min_value=0, value=0)
        feat10 = st.number_input("Dots in URL", min_value=0, value=1)
        feat11 = st.number_input("Hyphens Count", min_value=0, value=0)
        feat12 = st.number_input("Underscores Count", min_value=0, value=0)
        feat13 = st.number_input("Slashes Count", min_value=0, value=2)
        feat14 = st.number_input("Question Marks", min_value=0, value=0)
        feat15 = st.number_input("Equal Signs", min_value=0, value=0)
        feat16 = st.number_input("At Symbols (@)", min_value=0, value=0)
    with col4:
        feat17 = st.number_input("Ampersands (&)", min_value=0, value=0)
        feat18 = st.number_input("Percent Signs (%)", min_value=0, value=0)
        feat19 = st.number_input("Hash Signs (#)", min_value=0, value=0)
        feat20 = st.number_input("Digits Count", min_value=0, value=2)
        feat21 = st.number_input("Letters Count", min_value=0, value=20)
        feat22 = st.slider("Alexa Rank (normalized)", 0.0, 1.0, 0.5)
        feat23 = st.slider("Domain Trust Score", 0.0, 1.0, 0.8)
        feat24 = st.selectbox("Has SSL Certificate", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with col5:
        feat25 = st.number_input("Redirects Count", min_value=0, value=0)
        feat26 = st.slider("Page Load Time (seconds)", 0.0, 10.0, 2.0)
        feat27 = st.selectbox("Has Forms", [0, 1], format_func=lambda x: "Yes" if x else "No")
        feat28 = st.selectbox("Has Hidden Elements", [0, 1], format_func=lambda x: "Yes" if x else "No")
        feat29 = st.slider("External Links Ratio", 0.0, 1.0, 0.3)
        feat30 = st.slider("Image to Text Ratio", 0.0, 1.0, 0.5)

    def _manual_row():
        return [
            url_length, domain_age, subdomain_count, special_chars,
            https_usage, google_index, page_rank, domain_registration_length,
            feat9, feat10, feat11, feat12, feat13, feat14, feat15, feat16,
            feat17, feat18, feat19, feat20, feat21, feat22, feat23, feat24,
            feat25, feat26, feat27, feat28, feat29, feat30
        ]

    if st.button("🔍 Analyze URL", type="primary"):
        if model is None:
            st.error("Model not loaded"); st.stop()

        X_df = pd.DataFrame([_manual_row()], columns=EXPECTED)
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        if X_df.isna().any().any():
            st.error("Non-numeric inputs detected")
            st.write(X_df.isna().sum()); st.stop()

        if _uses_internal_scaler(model):
            X_in = X_df
        else:
            if scaler is None:
                st.error("External scaler missing for this model"); st.stop()
            if hasattr(scaler, "feature_names_in_"):
                missing = [c for c in scaler.feature_names_in_ if c not in X_df.columns]
                extra = [c for c in X_df.columns if c not in scaler.feature_names_in_]
                if missing or extra:
                    st.error(f"Scaler feature-name mismatch. missing={missing} extra={extra}"); st.stop()
            X_in = scaler.transform(X_df)

        proba = model.predict_proba(X_in)[0]
        pred = int(np.argmax(proba))

        st.header("🎯 Prediction Results")
        if pred == 1:
            st.error(f"⚠️ PHISHING DETECTED - Confidence: {proba[1]:.2%}")
        else:
            st.success(f"✅ LEGITIMATE URL - Confidence: {proba[0]:.2%}")
        c1, c2 = st.columns(2)
        with c1: st.metric("Legitimate Probability", f"{proba[0]:.2%}")
        with c2: st.metric("Phishing Probability", f"{proba[1]:.2%}")

# -----------------------------------------------------------------------------
# URL Analysis
# -----------------------------------------------------------------------------
elif input_method == "URL Analysis":
    st.header("URL Analysis")
    st.info("Enter a URL to extract features automatically (feature extraction not implemented).")
    url_input = st.text_input("Enter URL:", placeholder="https://example.com")
    if st.button("Analyze URL"):
        if url_input:
            st.warning("Feature extraction from URL not implemented yet. Use manual input or batch.")
        else:
            st.error("Please enter a URL")

# -----------------------------------------------------------------------------
# Batch Prediction (fixed)
# -----------------------------------------------------------------------------
elif input_method == "Batch Prediction":
    st.header("Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

    if uploaded_file is not None:
        # Read and normalize headers early
        df = pd.read_csv(uploaded_file)
        df.columns = (
            df.columns
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
        )

        st.write({"rows": len(df), "cols": len(df.columns)})
        st.write("Data preview:")
        st.dataframe(df.head())

        # If an obvious target is present, show a tally
        target_like = [c for c in df.columns if c.strip().lower() in {"class","label","target","y"}]
        if target_like:
            st.info(f"Found target-like columns: {target_like}. They will be dropped for prediction.")
            for tcol in target_like:
                try:
                    st.write({tcol: df[tcol].value_counts(dropna=False).to_dict()})
                except Exception:
                    pass

        st.write(f"Columns in file: {list(df.columns)}")
        st.write(f"Number of columns: {len(df.columns)}")

        if st.button("Run Batch Prediction", type="primary"):
            if model is None:
                st.error("Model not loaded"); st.stop()

            try:
                # Drop target-like columns
                drop_candidates = [c for c in df.columns if c.strip().lower() in {"class","label","target","y"}]
                features_df = df.drop(columns=drop_candidates, errors="ignore").copy()

                # Require exact set and order of features
                missing = [c for c in EXPECTED if c not in features_df.columns]
                extra   = [c for c in features_df.columns if c not in EXPECTED]
                if missing or extra:
                    st.error("❌ Columns mismatch")
                    st.write({"missing": missing, "extra": extra})
                    st.stop()

                # Reorder to training order
                features_df = features_df[EXPECTED]

                # Ensure numeric
                features_df = features_df.apply(pd.to_numeric, errors="coerce")
                if features_df.isna().any().any():
                    st.error("❌ Non-numeric values detected after coercion")
                    st.write(features_df.isna().sum())
                    st.stop()

                # Choose scaling path
                if _uses_internal_scaler(model):
                    X_in = features_df
                    st.info("Using model’s internal scaler")
                else:
                    if scaler is None:
                        st.error("External scaler is required but missing"); st.stop()
                    if hasattr(scaler, "feature_names_in_"):
                        if list(scaler.feature_names_in_) != list(EXPECTED):
                            st.error("Scaler feature-name order does not match EXPECTED"); st.stop()
                    X_in = scaler.transform(features_df)
                    st.success("Features scaled using external scaler")

                # Predict
                predictions = model.predict(X_in)
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_in)
                    legit_prob = probabilities[:, 0]
                    phish_prob = probabilities[:, 1]
                else:
                    # Fallback if model lacks predict_proba
                    phish_prob = predictions.astype(float)
                    legit_prob = 1.0 - phish_prob

                # Assemble results
                results_df = df.copy()
                results_df["Prediction"] = predictions
                results_df["Legitimate_Prob"] = legit_prob
                results_df["Phishing_Prob"] = phish_prob
                results_df["Status"] = results_df["Prediction"].map({0: "Legitimate", 1: "Phishing"})

                st.success("✅ Batch prediction completed")

                st.subheader("📊 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Total URLs", len(results_df))
                with col2: st.metric("Legitimate", int((predictions == 0).sum()))
                with col3: st.metric("Phishing", int((predictions == 1).sum()))

                display_columns = ["Status", "Legitimate_Prob", "Phishing_Prob"]
                if target_like:
                    # show first target-like column if present
                    display_columns = [target_like[0]] + display_columns

                st.subheader("🔍 Detailed Results")
                st.dataframe(results_df[display_columns])

                # Optional quick accuracy vs one binary target column if present
                for tcol in target_like:
                    try:
                        uniq = set(pd.Series(df[tcol]).dropna().unique())
                        if uniq <= {0,1}:
                            acc = (predictions == df[tcol].to_numpy()).mean()
                            st.metric(f"Accuracy vs '{tcol}'", f"{acc:.1%}")
                            break
                    except Exception:
                        pass

                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="phishing_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error in batch prediction: {e}")
                st.write("**Troubleshooting:**")
                st.write("1) Headers must exactly match EXPECTED.")
                st.write("2) Remove any target column (class/label/target/y).")
                st.write("3) All values must be numeric.")
                st.write("4) Scaler and model must come from the same training pipeline.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("Built by Group AJ 🎈 | Cybersecurity DLI Project")