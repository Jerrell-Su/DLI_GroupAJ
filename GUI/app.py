# app.py — Streamlit UI for model.pkl + scaler.pkl
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
st.set_page_config(page_title="URL Phishing Detection", page_icon="🔒", layout="wide")
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
# Load backup model (RandomForest on SCALED features) + external scaler
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
        st.error("scaler.pkl not found (required for model.pkl)")
        return None
    return joblib.load(p)

model = load_model()
scaler = load_scaler()

def _uses_internal_scaler(_):
    # model is a plain RF, so False
    return False

# -----------------------------------------------------------------------------
# Canonical training feature names
# Prefer scaler.feature_names_in_ to guarantee exact match for transform()
# -----------------------------------------------------------------------------
def _resolve_expected(scaler):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # fallback: manual list of 30 features (must match your training CSV headers)
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

EXPECTED = _resolve_expected(scaler)

# Diagnostics
DEBUG = False  # set True when you want to see developer info
if DEBUG:
    st.caption(f"len(EXPECTED)={len(EXPECTED)}")
    if hasattr(scaler, "feature_names_in_"):
        st.caption(f"Scaler expects: {list(scaler.feature_names_in_)}")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["URL Analysis", "Batch Prediction"]
)

# -----------------------------------------------------------------------------
# URL Analysis
# -----------------------------------------------------------------------------
if input_method == "URL Analysis":
    st.header("🔍 Single URL Analysis")
    st.info("Enter a URL to extract features automatically and detect phishing.")
    
    url_input = st.text_input("Enter URL:", placeholder="https://example.com")
    
    if st.button("🔍 Analyze URL", type="primary"):
        if not url_input:
            st.error("Please enter a URL")
            st.stop()
            
        if model is None or scaler is None:
            st.error("Model or scaler not loaded")
            st.stop()
        
        # Note: Feature extraction would go here
        st.warning("⚠️ URL feature extraction is not yet implemented.")
        st.markdown("""
        **To implement URL feature extraction, you would need to:**
        
        1. Parse the URL components (domain, path, query parameters, etc.)
        2. Extract the 30 features your model expects:
           - url_length, domain_age, subdomain_count, special_chars
           - https_usage, google_index, page_rank, domain_registration_length
           - And 22 other features...
        3. Create a DataFrame with these features
        4. Apply the scaler and make predictions
        
        **For now, please use the "Batch Prediction" option with a CSV file containing the extracted features.**
        """)

# -----------------------------------------------------------------------------
# Batch Prediction (strict to match scaler + RF)
# -----------------------------------------------------------------------------
elif input_method == "Batch Prediction":
    st.header("📊 Batch Prediction")
    st.info("Upload a CSV file with extracted URL features for batch analysis.")
    
    uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

        st.write({"rows": len(df), "cols": len(df.columns)})
        st.write("**Data preview:**")
        st.dataframe(df.head())

        target_like = [c for c in df.columns if c.strip().lower() in {"class","label","target","y"}]
        if target_like:
            st.info(f"Found target-like columns: {target_like}. They will be dropped for prediction.")
            for tcol in target_like:
                try:
                    st.write({tcol: df[tcol].value_counts(dropna=False).to_dict()})
                except Exception:
                    pass

        st.write(f"**Columns in file:** {list(df.columns)}")
        st.write(f"**Number of columns:** {len(df.columns)}")
        st.write(f"**Expected columns:** {EXPECTED}")

        if st.button("🚀 Run Batch Prediction", type="primary"):
            if model is None or scaler is None:
                st.error("Model or scaler not loaded")
                st.stop()

            try:
                # Drop target-like columns
                features_df = df.drop(columns=target_like, errors="ignore").copy()

                # Require exact set and order of features
                missing = [c for c in EXPECTED if c not in features_df.columns]
                extra   = [c for c in features_df.columns if c not in EXPECTED]
                if missing or extra:
                    st.error("❌ Columns mismatch")
                    st.write({"missing": missing, "extra": extra})
                    st.stop()

                features_df = features_df[EXPECTED]

                # Ensure numeric
                features_df = features_df.apply(pd.to_numeric, errors="coerce")
                if features_df.isna().any().any():
                    st.error("❌ Non-numeric values detected after coercion")
                    st.write(features_df.isna().sum())
                    st.stop()

                # External scaler path
                if hasattr(scaler, "feature_names_in_"):
                    if list(scaler.feature_names_in_) != list(EXPECTED):
                        st.error("Scaler feature-name order does not match EXPECTED")
                        st.stop()
                X_in = scaler.transform(features_df)

                # Predict
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_in)
                    predictions = np.argmax(probabilities, axis=1)
                    legit_prob = probabilities[:, 0]
                    phish_prob = probabilities[:, 1]
                else:
                    predictions = model.predict(X_in)
                    phish_prob = predictions.astype(float)
                    legit_prob = 1.0 - phish_prob

                # Assemble results
                results_df = df.copy()
                results_df["Prediction"] = predictions
                results_df["Legitimate_Prob"] = legit_prob
                results_df["Phishing_Prob"] = phish_prob
                results_df["Status"] = results_df["Prediction"].map({0: "Legitimate", 1: "Phishing"})

                st.success("✅ Batch prediction completed successfully!")

                st.subheader("📊 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                with col1: 
                    st.metric("Total URLs", len(results_df))
                with col2: 
                    legit_count = int((predictions == 0).sum())
                    st.metric("Legitimate", legit_count, delta=f"{legit_count/len(results_df)*100:.1f}%")
                with col3: 
                    phish_count = int((predictions == 1).sum())
                    st.metric("Phishing", phish_count, delta=f"{phish_count/len(results_df)*100:.1f}%")

                display_columns = ["Status", "Legitimate_Prob", "Phishing_Prob"]
                if target_like:
                    display_columns = [target_like[0]] + display_columns

                st.subheader("🔍 Detailed Results")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox("Filter by Status:", ["All", "Legitimate", "Phishing"])
                with col2:
                    confidence_threshold = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0, 0.1)
                
                # Apply filters
                filtered_df = results_df.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df["Status"] == status_filter]
                
                if confidence_threshold > 0:
                    filtered_df = filtered_df[
                        (filtered_df["Legitimate_Prob"] >= confidence_threshold) | 
                        (filtered_df["Phishing_Prob"] >= confidence_threshold)
                    ]
                
                st.dataframe(filtered_df[display_columns], use_container_width=True)

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
                    label="📥 Download Complete Results CSV",
                    data=csv,
                    file_name="phishing_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error in batch prediction: {e}")
                st.markdown("**Troubleshooting:**")
                st.markdown("1. Headers must exactly match the expected feature names")
                st.markdown("2. Remove any target column (class/label/target/y)")
                st.markdown("3. All values must be numeric")
                st.markdown("4. scaler.pkl must match model.pkl training run")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("Built by Group AJ 🎈 | Cybersecurity DLI Project")