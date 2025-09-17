# # app.py — Streamlit UI for model.pkl + scaler.pkl with integrated feature extraction
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import warnings
# from pathlib import Path
# import sys
# import os

# # Add the current directory to Python path to import feature.py
# current_dir = Path(__file__).resolve().parent
# sys.path.append(str(current_dir))
# sys.path.append(str(current_dir.parent))  # Also add parent directory

# try:
#     from feature import FeatureExtraction
#     FEATURE_EXTRACTION_AVAILABLE = True
# except ImportError as e:
#     # Try alternative import methods
#     try:
#         import importlib.util
#         spec = importlib.util.spec_from_file_location("feature", current_dir / "feature.py")
#         feature_module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(feature_module)
#         FeatureExtraction = feature_module.FeatureExtraction
#         FEATURE_EXTRACTION_AVAILABLE = True
#     except Exception as e2:
#         st.warning(f"Feature extraction not available: {e}")
#         st.caption("Make sure feature.py is in the same directory as app.py and all dependencies are installed:")
#         st.code("pip install beautifulsoup4 requests python-whois googlesearch-python python-dateutil lxml")
#         FEATURE_EXTRACTION_AVAILABLE = False

# warnings.filterwarnings("ignore")

# # -----------------------------------------------------------------------------
# # Page configuration
# # -----------------------------------------------------------------------------
# st.set_page_config(page_title="URL Phishing Detection", page_icon="🔒", layout="wide")
# st.title("🔒 URL-Based Phishing Detection System")
# st.markdown("**Detect malicious URLs using machine learning**")

# # -----------------------------------------------------------------------------
# # Paths
# # -----------------------------------------------------------------------------
# APP_DIR = Path(__file__).resolve().parent
# ROOT_DIR = APP_DIR.parent
# MODEL_DIR = ROOT_DIR / "Main_Model"

# def _pick(*cands):
#     for p in cands:
#         if p and p.is_file():
#             return p
#     return None

# # -----------------------------------------------------------------------------
# # Load backup model (RandomForest on SCALED features) + external scaler
# # -----------------------------------------------------------------------------
# @st.cache_resource
# def load_model():
#     p = _pick(MODEL_DIR/"model.pkl", APP_DIR/"model.pkl", ROOT_DIR/"model.pkl")
#     if not p:
#         st.error("model.pkl not found")
#         st.caption(f"Tried: {[str(x) for x in [MODEL_DIR/'model.pkl', APP_DIR/'model.pkl', ROOT_DIR/'model.pkl']]}")
#         return None
#     return joblib.load(p)

# @st.cache_resource
# def load_scaler():
#     p = _pick(MODEL_DIR/"scaler.pkl", APP_DIR/"scaler.pkl", ROOT_DIR/"scaler.pkl")
#     if not p:
#         st.error("scaler.pkl not found (required for model.pkl)")
#         return None
#     return joblib.load(p)

# model = load_model()
# scaler = load_scaler()

# def _uses_internal_scaler(_):
#     # model is a plain RF, so False
#     return False

# # -----------------------------------------------------------------------------
# # Feature extraction function
# # -----------------------------------------------------------------------------
# def extract_url_features(url):
#     """Extract features from URL using the FeatureExtraction class"""
#     if not FEATURE_EXTRACTION_AVAILABLE:
#         raise ImportError("Feature extraction module not available")
    
#     try:
#         # Initialize feature extractor
#         extractor = FeatureExtraction(url)
        
#         # Get the features list
#         features = extractor.getFeaturesList()
        
#         # Define feature names based on the FeatureExtraction class methods
#         feature_names = [
#             'UsingIp', 'longUrl', 'shortUrl', 'symbol', 'redirecting',
#             'prefixSuffix', 'SubDomains', 'Hppts', 'DomainRegLen', 'Favicon',
#             'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL', 'LinksInScriptTags',
#             'ServerFormHandler', 'InfoEmail', 'AbnormalURL', 'WebsiteForwarding', 'StatusBarCust',
#             'DisableRightClick', 'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain', 'DNSRecording',
#             'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage', 'StatsReport'
#         ]
        
#         # Create DataFrame with extracted features
#         feature_df = pd.DataFrame([features], columns=feature_names)
        
#         return feature_df, None  # Return DataFrame and no error
        
#     except Exception as e:
#         return None, str(e)

# # -----------------------------------------------------------------------------
# # Canonical training feature names
# # Prefer scaler.feature_names_in_ to guarantee exact match for transform()
# # -----------------------------------------------------------------------------
# def _resolve_expected(scaler):
#     if hasattr(scaler, "feature_names_in_"):
#         return list(scaler.feature_names_in_)
#     # fallback: manual list of 30 features (must match your training CSV headers)
#     return [
#         "url_length","domain_age","subdomain_count","special_chars",
#         "https_usage","google_index","page_rank","domain_registration_length",
#         "suspicious_keywords","dots_count","hyphens_count","underscores_count",
#         "slashes_count","question_marks","equal_signs","at_symbols",
#         "ampersands","percent_signs","hash_signs","digits_count",
#         "letters_count","alexa_rank","domain_trust","ssl_certificate",
#         "redirects_count","page_load_time","has_forms","hidden_elements",
#         "external_links_ratio","image_text_ratio"
#     ]

# EXPECTED = _resolve_expected(scaler)

# # Diagnostics
# DEBUG = False  # set True when you want to see developer info
# if DEBUG:
#     st.caption(f"len(EXPECTED)={len(EXPECTED)}")
#     if hasattr(scaler, "feature_names_in_"):
#         st.caption(f"Scaler expects: {list(scaler.feature_names_in_)}")

# # -----------------------------------------------------------------------------
# # Sidebar
# # -----------------------------------------------------------------------------
# st.sidebar.header("Input Method")
# input_method = st.sidebar.radio(
#     "Choose input method:",
#     ["URL Analysis", "Batch Prediction"]
# )

# # -----------------------------------------------------------------------------
# # URL Analysis
# # -----------------------------------------------------------------------------
# if input_method == "URL Analysis":
#     st.header("🔍 Single URL Analysis")
#     st.info("Enter a URL to extract features automatically and detect phishing.")
    
#     url_input = st.text_input("Enter URL:", placeholder="https://example.com")
    
#     if st.button("🔍 Analyze URL", type="primary"):
#         if not url_input:
#             st.error("Please enter a URL")
#             st.stop()
            
#         if model is None or scaler is None:
#             st.error("Model or scaler not loaded")
#             st.stop()
            
#         if not FEATURE_EXTRACTION_AVAILABLE:
#             st.error("Feature extraction module not available. Please ensure feature.py is in the correct location.")
#             st.stop()
        
#         # Add protocol if missing
#         if not url_input.startswith(('http://', 'https://')):
#             url_input = 'https://' + url_input
        
#         with st.spinner("Extracting features from URL..."):
#             try:
#                 # Extract features
#                 features_df, error = extract_url_features(url_input)
                
#                 if error:
#                     st.error(f"Error extracting features: {error}")
#                     st.stop()
                
#                 st.success("✅ Features extracted successfully!")
                
#                 # Display extracted features
#                 with st.expander("View Extracted Features"):
#                     st.dataframe(features_df)
                
#                 # Debug info
#                 st.info(f"Extracted {len(features_df.columns)} features, model expects {len(EXPECTED)}")
                
#                 # Force the features to match exactly what the model expects
#                 if len(features_df.columns) != len(EXPECTED):
#                     st.warning(f"Adjusting feature count from {len(features_df.columns)} to {len(EXPECTED)}")
                    
#                     # Get the feature values as a list
#                     feature_values = features_df.iloc[0].tolist()
                    
#                     # Adjust the feature count
#                     if len(feature_values) < len(EXPECTED):
#                         # Add zeros for missing features
#                         feature_values.extend([0] * (len(EXPECTED) - len(feature_values)))
#                         st.info(f"Added {len(EXPECTED) - len(features_df.columns)} padding features")
#                     elif len(feature_values) > len(EXPECTED):
#                         # Trim extra features
#                         feature_values = feature_values[:len(EXPECTED)]
#                         st.info(f"Trimmed to first {len(EXPECTED)} features")
                    
#                     # Create new DataFrame with correct feature count and names
#                     features_df = pd.DataFrame([feature_values], columns=EXPECTED)
#                     st.success(f"✅ Features adjusted to match model expectations ({len(EXPECTED)} features)")
                
#                 # Convert to numpy array for prediction
#                 X_features = features_df.values
                
#                 # Apply scaler
#                 try:
#                     X_scaled = scaler.transform(X_features)
#                 except Exception as e:
#                     st.error(f"Error applying scaler: {e}")
#                     st.error("This might be due to feature format issues.")
#                     st.error("Debug info:")
#                     st.write(f"Features shape: {X_features.shape}")
#                     st.write(f"Features DataFrame columns: {len(features_df.columns)}")
#                     if hasattr(scaler, "n_features_in_"):
#                         st.write(f"Scaler expects: {scaler.n_features_in_} features")
#                     st.stop()
                
#                 # Make prediction
#                 try:
#                     if hasattr(model, "predict_proba"):
#                         probabilities = model.predict_proba(X_scaled)
#                         prediction = np.argmax(probabilities, axis=1)[0]
#                         legit_prob = probabilities[0][0]
#                         phish_prob = probabilities[0][1]
#                     else:
#                         prediction = model.predict(X_scaled)[0]
#                         phish_prob = float(prediction)
#                         legit_prob = 1.0 - phish_prob
                    
#                     # Display results
#                     st.subheader("🎯 Prediction Results")
                    
#                     col1, col2, col3 = st.columns(3)
                    
#                     with col1:
#                         status = "🔴 Phishing" if prediction == 1 else "🟢 Legitimate"
#                         st.metric("Status", status)
                    
#                     with col2:
#                         st.metric("Legitimate Probability", f"{legit_prob:.2%}")
                    
#                     with col3:
#                         st.metric("Phishing Probability", f"{phish_prob:.2%}")
                    
#                     # Confidence indicator
#                     confidence = max(legit_prob, phish_prob)
#                     if confidence > 0.8:
#                         confidence_text = "🟢 High Confidence"
#                     elif confidence > 0.6:
#                         confidence_text = "🟡 Medium Confidence"
#                     else:
#                         confidence_text = "🔴 Low Confidence"
                    
#                     st.info(f"Prediction Confidence: {confidence_text} ({confidence:.1%})")
                    
#                     # Additional analysis
#                     st.subheader("📊 Detailed Analysis")
                    
#                     if prediction == 1:  # Phishing
#                         st.error("⚠️ **Warning: This URL appears to be malicious!**")
#                         st.markdown("""
#                         **Recommendations:**
#                         - Do not enter personal information
#                         - Do not download files from this site
#                         - Verify the URL with the legitimate organization
#                         - Report this URL to security authorities
#                         """)
#                     else:  # Legitimate
#                         st.success("✅ **This URL appears to be legitimate**")
#                         st.markdown("""
#                         **Note:**
#                         - This analysis is based on URL characteristics only
#                         - Always exercise caution when sharing personal information
#                         - Verify SSL certificates and site authenticity
#                         """)
                    
#                 except Exception as e:
#                     st.error(f"Error making prediction: {e}")
                    
#             except Exception as e:
#                 st.error(f"Unexpected error during analysis: {e}")
#                 st.markdown("**Troubleshooting tips:**")
#                 st.markdown("- Check if the URL is accessible")
#                 st.markdown("- Ensure you have internet connectivity")
#                 st.markdown("- Try with a different URL format")

# # -----------------------------------------------------------------------------
# # Batch Prediction (strict to match scaler + RF)
# # -----------------------------------------------------------------------------
# elif input_method == "Batch Prediction":
#     st.header("📊 Batch Prediction")
#     st.info("Upload a CSV file with extracted URL features for batch analysis.")
    
#     uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

#         st.write({"rows": len(df), "cols": len(df.columns)})
#         st.write("**Data preview:**")
#         st.dataframe(df.head())

#         target_like = [c for c in df.columns if c.strip().lower() in {"class","label","target","y"}]
#         if target_like:
#             st.info(f"Found target-like columns: {target_like}. They will be dropped for prediction.")
#             for tcol in target_like:
#                 try:
#                     st.write({tcol: df[tcol].value_counts(dropna=False).to_dict()})
#                 except Exception:
#                     pass

#         st.write(f"**Columns in file:** {list(df.columns)}")
#         st.write(f"**Number of columns:** {len(df.columns)}")
#         st.write(f"**Expected columns:** {EXPECTED}")

#         if st.button("🚀 Run Batch Prediction", type="primary"):
#             if model is None or scaler is None:
#                 st.error("Model or scaler not loaded")
#                 st.stop()

#             try:
#                 # Drop target-like columns
#                 features_df = df.drop(columns=target_like, errors="ignore").copy()

#                 # Require exact set and order of features
#                 missing = [c for c in EXPECTED if c not in features_df.columns]
#                 extra   = [c for c in features_df.columns if c not in EXPECTED]
#                 if missing or extra:
#                     st.error("❌ Columns mismatch")
#                     st.write({"missing": missing, "extra": extra})
#                     st.stop()

#                 features_df = features_df[EXPECTED]

#                 # Ensure numeric
#                 features_df = features_df.apply(pd.to_numeric, errors="coerce")
#                 if features_df.isna().any().any():
#                     st.error("❌ Non-numeric values detected after coercion")
#                     st.write(features_df.isna().sum())
#                     st.stop()

#                 # External scaler path
#                 if hasattr(scaler, "feature_names_in_"):
#                     if list(scaler.feature_names_in_) != list(EXPECTED):
#                         st.error("Scaler feature-name order does not match EXPECTED")
#                         st.stop()
#                 X_in = scaler.transform(features_df)

#                 # Predict
#                 if hasattr(model, "predict_proba"):
#                     probabilities = model.predict_proba(X_in)
#                     predictions = np.argmax(probabilities, axis=1)
#                     legit_prob = probabilities[:, 0]
#                     phish_prob = probabilities[:, 1]
#                 else:
#                     predictions = model.predict(X_in)
#                     phish_prob = predictions.astype(float)
#                     legit_prob = 1.0 - phish_prob

#                 # Assemble results
#                 results_df = df.copy()
#                 results_df["Prediction"] = predictions
#                 results_df["Legitimate_Prob"] = legit_prob
#                 results_df["Phishing_Prob"] = phish_prob
#                 results_df["Status"] = results_df["Prediction"].map({0: "Legitimate", 1: "Phishing"})

#                 st.success("✅ Batch prediction completed successfully!")

#                 st.subheader("📊 Prediction Summary")
#                 col1, col2, col3 = st.columns(3)
#                 with col1: 
#                     st.metric("Total URLs", len(results_df))
#                 with col2: 
#                     legit_count = int((predictions == 0).sum())
#                     st.metric("Legitimate", legit_count, delta=f"{legit_count/len(results_df)*100:.1f}%")
#                 with col3: 
#                     phish_count = int((predictions == 1).sum())
#                     st.metric("Phishing", phish_count, delta=f"{phish_count/len(results_df)*100:.1f}%")

#                 display_columns = ["Status", "Legitimate_Prob", "Phishing_Prob"]
#                 if target_like:
#                     display_columns = [target_like[0]] + display_columns

#                 st.subheader("🔍 Detailed Results")
                
#                 # Filter options
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     status_filter = st.selectbox("Filter by Status:", ["All", "Legitimate", "Phishing"])
#                 with col2:
#                     confidence_threshold = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0, 0.1)
                
#                 # Apply filters
#                 filtered_df = results_df.copy()
#                 if status_filter != "All":
#                     filtered_df = filtered_df[filtered_df["Status"] == status_filter]
                
#                 if confidence_threshold > 0:
#                     filtered_df = filtered_df[
#                         (filtered_df["Legitimate_Prob"] >= confidence_threshold) | 
#                         (filtered_df["Phishing_Prob"] >= confidence_threshold)
#                     ]
                
#                 st.dataframe(filtered_df[display_columns], use_container_width=True)

#                 # Optional quick accuracy vs one binary target column if present
#                 for tcol in target_like:
#                     try:
#                         uniq = set(pd.Series(df[tcol]).dropna().unique())
#                         if uniq <= {0,1}:
#                             acc = (predictions == df[tcol].to_numpy()).mean()
#                             st.metric(f"Accuracy vs '{tcol}'", f"{acc:.1%}")
#                             break
#                     except Exception:
#                         pass

#                 # Download
#                 csv = results_df.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Complete Results CSV",
#                     data=csv,
#                     file_name="phishing_predictions.csv",
#                     mime="text/csv",
#                 )

#             except Exception as e:
#                 st.error(f"❌ Error in batch prediction: {e}")
#                 st.markdown("**Troubleshooting:**")
#                 st.markdown("1. Headers must exactly match the expected feature names")
#                 st.markdown("2. Remove any target column (class/label/target/y)")
#                 st.markdown("3. All values must be numeric")
#                 st.markdown("4. scaler.pkl must match model.pkl training run")

# # -----------------------------------------------------------------------------
# # Footer
# # -----------------------------------------------------------------------------
# st.markdown("---")
# st.markdown("Built by Group AJ 🎈 | Cybersecurity DLI Project")

# # -----------------------------------------------------------------------------
# # Additional Information Section
# # -----------------------------------------------------------------------------
# with st.sidebar:
#     st.markdown("---")
#     st.header("ℹ️ Information")
    
#     st.markdown("**Feature Extraction Status:**")
#     if FEATURE_EXTRACTION_AVAILABLE:
#         st.success("✅ Available")
#     else:
#         st.error("❌ Not Available")
#         st.caption("Make sure feature.py is in the correct location")
    
#     st.markdown("**Model Status:**")
#     if model is not None:
#         st.success("✅ Loaded")
#     else:
#         st.error("❌ Not Loaded")
    
#     st.markdown("**Scaler Status:**")
#     if scaler is not None:
#         st.success("✅ Loaded")
#     else:
#         st.error("❌ Not Loaded")
    
#     with st.expander("🔧 Technical Details"):
#         st.markdown(f"**Expected Features:** {len(EXPECTED)}")
#         st.markdown(f"**Feature Extraction Available:** {FEATURE_EXTRACTION_AVAILABLE}")
#         if hasattr(scaler, "feature_names_in_"):
#             st.markdown(f"**Scaler Features:** {len(scaler.feature_names_in_)}")
# #

# app.py — Streamlit UI for model.pkl + scaler.pkl with integrated feature extraction
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue

# -------------------------------------------------------------------
# Import feature extractor
# -------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

try:
    from feature import FeatureExtraction
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError as e:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("feature", current_dir / "feature.py")
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)
        FeatureExtraction = feature_module.FeatureExtraction
        FEATURE_EXTRACTION_AVAILABLE = True
    except Exception as e2:
        st.warning(f"Feature extraction not available: {e}")
        st.caption("Make sure feature.py is in the same directory as app.py and dependencies installed:")
        st.code("pip install beautifulsoup4 requests python-whois googlesearch-python python-dateutil lxml")
        FEATURE_EXTRACTION_AVAILABLE = False

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(page_title="URL Phishing Detection", page_icon="🔒", layout="wide")
st.title("🔒 URL-Based Phishing Detection System")
st.markdown("**Detect malicious URLs using machine learning**")

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR = ROOT_DIR / "Main_Model"

def _pick(*cands):
    for p in cands:
        if p and p.is_file():
            return p
    return None

# -------------------------------------------------------------------
# Load model and scaler
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    p = _pick(MODEL_DIR/"model.pkl", APP_DIR/"model.pkl", ROOT_DIR/"model.pkl")
    if not p:
        st.error("model.pkl not found")
        return None
    return joblib.load(p)

@st.cache_resource
def load_scaler():
    p = _pick(MODEL_DIR/"scaler.pkl", APP_DIR/"scaler.pkl", ROOT_DIR/"scaler.pkl")
    if not p:
        st.error("scaler.pkl not found")
        return None
    return joblib.load(p)

model = load_model()
scaler = load_scaler()

# -------------------------------------------------------------------
# Feature extraction wrapper
# -------------------------------------------------------------------
def extract_url_features(url):
    if not FEATURE_EXTRACTION_AVAILABLE:
        raise ImportError("Feature extraction not available")
    extractor = FeatureExtraction(url)
    features = extractor.getFeaturesList()
    feature_names = [
        'UsingIP','LongURL','ShortURL','Symbol@','Redirecting//',
        'PrefixSuffix-','SubDomains','HTTPS','DomainRegLen','Favicon',
        'NonStdPort','HTTPSDomainURL','RequestURL','AnchorURL','LinksInScriptTags',
        'ServerFormHandler','InfoEmail','AbnormalURL','WebsiteForwarding',
        'StatusBarCust','DisableRightClick','UsingPopupWindow','IframeRedirection',
        'AgeofDomain','DNSRecording','WebsiteTraffic','PageRank','GoogleIndex',
        'LinksPointingToPage','StatsReport'
    ]
    return pd.DataFrame([features], columns=feature_names), None

# -------------------------------------------------------------------
# Training feature names
# -------------------------------------------------------------------
def _resolve_expected(scaler):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return [
        'UsingIP','LongURL','ShortURL','Symbol@','Redirecting//',
        'PrefixSuffix-','SubDomains','HTTPS','DomainRegLen','Favicon',
        'NonStdPort','HTTPSDomainURL','RequestURL','AnchorURL','LinksInScriptTags',
        'ServerFormHandler','InfoEmail','AbnormalURL','WebsiteForwarding',
        'StatusBarCust','DisableRightClick','UsingPopupWindow','IframeRedirection',
        'AgeofDomain','DNSRecording','WebsiteTraffic','PageRank','GoogleIndex',
        'LinksPointingToPage','StatsReport'
    ]

EXPECTED = _resolve_expected(scaler)

# -------------------------------------------------------------------
# Shared processor with timeout + diagnostics
# -------------------------------------------------------------------
def process_url(url, timeout=10):
    if not str(url).startswith(("http://", "https://")):
        url = "https://" + str(url)

    def _worker(u, q):
        try:
            fdf, _ = extract_url_features(u)
            feats = fdf.iloc[0].tolist()
            q.put((feats, None))
        except Exception as e:
            q.put(([0] * len(EXPECTED), str(e)))

    q = Queue()
    p = Process(target=_worker, args=(url, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return [0] * len(EXPECTED), f"Timed out after {timeout}s"
    if not q.empty():
        return q.get()
    return [0] * len(EXPECTED), "Unknown error"

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["URL Analysis", "Batch Prediction"])

# -------------------------------------------------------------------
# Single URL Analysis
# -------------------------------------------------------------------
if input_method == "URL Analysis":
    st.header("🔍 Single URL Analysis")
    url_input = st.text_input("Enter URL:", placeholder="https://example.com")

    if st.button("🔍 Analyze URL", type="primary"):
        if not url_input:
            st.error("Please enter a URL"); st.stop()
        if model is None or scaler is None:
            st.error("Model or scaler not loaded"); st.stop()
        if not FEATURE_EXTRACTION_AVAILABLE:
            st.error("Feature extraction not available"); st.stop()

        with st.spinner("Extracting features..."):
            feats, err = process_url(url_input, timeout=10)
            features_df = pd.DataFrame([feats], columns=EXPECTED)

        st.subheader("🔍 Extracted Features")
        st.write(features_df)
        if all(v == 0 for v in feats):
            st.error("⚠️ All features are zero — model input invalid")
            if err:
                st.warning(f"Reason: {err}")

        X_scaled = scaler.transform(features_df)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)
            pred = np.argmax(probs, axis=1)[0]
            legit_prob, phish_prob = probs[0][0], probs[0][1]
        else:
            pred = model.predict(X_scaled)[0]
            phish_prob = float(pred); legit_prob = 1.0 - phish_prob

        status = "🔴 Phishing" if pred==1 else "🟢 Legitimate"
        st.metric("Status", status)
        st.metric("Legitimate Probability", f"{legit_prob:.2%}")
        st.metric("Phishing Probability", f"{phish_prob:.2%}")

# -------------------------------------------------------------------
# Batch Prediction
# -------------------------------------------------------------------
elif input_method == "Batch Prediction":
    st.header("📊 Batch Prediction")
    st.info("Upload any file. All valid URLs inside will be extracted and analyzed.")

    uploaded_file = st.file_uploader("Upload file")

    URL_REGEX = re.compile(
        r'((?:https?://|http://|www\.)[^\s]+|[a-zA-Z0-9\-\.]+\.[a-z]{2,}(?:[^\s]*)?)',
        re.IGNORECASE
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        text = raw_bytes.decode(errors="ignore")
        urls = list(dict.fromkeys(URL_REGEX.findall(text)))
        if not urls:
            st.error("No valid URLs found in file"); st.stop()

        st.success(f"Found {len(urls)} URLs in file")
        st.dataframe(pd.DataFrame({"url": urls}).head())

        if st.button("🚀 Run Batch Prediction", type="primary"):
            if model is None or scaler is None:
                st.error("Model or scaler not loaded"); st.stop()
            if not FEATURE_EXTRACTION_AVAILABLE:
                st.error("Feature extraction not available"); st.stop()

            all_features, errors = [], []
            progress = st.progress(0)
            status = st.empty()

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_url, url, 10): idx for idx, url in enumerate(urls)}
                completed = 0
                for future in as_completed(futures):
                    feats, err = future.result()
                    all_features.append(feats)
                    errors.append(err)
                    completed += 1
                    status.text(f"Processed {completed}/{len(urls)} URLs")
                    progress.progress(completed/len(urls))

            progress.empty(); status.empty()

            features_df = pd.DataFrame(all_features, columns=EXPECTED)
            st.subheader("🔍 Extracted Features (first few)")
            st.write(features_df.head())

            zero_rows = features_df[(features_df == 0).all(axis=1)]
            if not zero_rows.empty:
                st.error(f"{len(zero_rows)} URLs produced all-zero features")
                st.dataframe(pd.DataFrame({
                    "url": [urls[i] for i, f in enumerate(all_features) if all(v==0 for v in f)],
                    "error": [errors[i] for i, f in enumerate(all_features) if all(v==0 for v in f)]
                }))

            X = scaler.transform(features_df)
            if hasattr(model,"predict_proba"):
                probs = model.predict_proba(X)
                preds = np.argmax(probs, axis=1)
                legit, phish = probs[:,0], probs[:,1]
            else:
                preds = model.predict(X)
                phish = preds.astype(float); legit = 1.0 - phish

            results = pd.DataFrame({
                "url": urls,
                "Prediction": preds,
                "Legitimate_Prob": legit,
                "Phishing_Prob": phish
            })
            results["Status"] = results["Prediction"].map({0:"Legitimate",1:"Phishing"})

            st.success("✅ Batch prediction complete")
            st.dataframe(results.head())

            csv = results.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "phishing_predictions.csv","text/csv")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("Built by Group AJ 🎈 | Cybersecurity DLI Project")

# -----------------------------------------------------------------------------
# Additional Information Section
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.header("ℹ️ Information")
    
    st.markdown("**Feature Extraction Status:**")
    if FEATURE_EXTRACTION_AVAILABLE:
        st.success("✅ Available")
    else:
        st.error("❌ Not Available")
        st.caption("Make sure feature.py is in the correct location")
    
    st.markdown("**Model Status:**")
    if model is not None:
        st.success("✅ Loaded")
    else:
        st.error("❌ Not Loaded")
    
    st.markdown("**Scaler Status:**")
    if scaler is not None:
        st.success("✅ Loaded")
    else:
        st.error("❌ Not Loaded")
    
    with st.expander("🔧 Technical Details"):
        st.markdown(f"**Expected Features:** {len(EXPECTED)}")
        st.markdown(f"**Feature Extraction Available:** {FEATURE_EXTRACTION_AVAILABLE}")
        if hasattr(scaler, "feature_names_in_"):
            st.markdown(f"**Scaler Features:** {len(scaler.feature_names_in_)}")











