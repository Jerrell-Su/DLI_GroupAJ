import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="URL Phishing Detection",
    page_icon="🔒",
    layout="wide"
)

# Title and description
st.title("🔒 URL-Based Phishing Detection System")
st.markdown("**Detect malicious URLs using machine learning**")

# Load your trained model (you'll need to save it first)
@st.cache_resource
def load_model():
    try:
        # Try different possible model file names with correct path
        model = joblib.load('../Main_Model/model.pkl')  # Path to Main_Model folder
        return model
    except:
        try:
            model = joblib.load('./model.pkl')  # Try local folder
            return model
        except:
            st.error("Model file not found. Please ensure 'model.pkl' is in Main_Model folder.")
            return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('../Main_Model/scaler.pkl')  # Path to Main_Model folder
        return scaler
    except:
        try:
            scaler = joblib.load('./scaler.pkl')  # Try local folder
            return scaler
        except:
            st.warning("Scaler not found. Using default StandardScaler.")
            return StandardScaler()

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Sidebar for input method
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Feature Input", "URL Analysis", "Batch Prediction"]
)

if input_method == "Manual Feature Input":
    st.header("Manual Feature Input")
    
    # Create columns for better layout
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
    
    # Additional features - 30 total features needed
    st.subheader("Additional URL Features (22 more needed)")
    col3, col4, col5 = st.columns(3)
    
    # Features 9-16
    with col3:
        feat9 = st.number_input("Suspicious Keywords Count", min_value=0, value=0)
        feat10 = st.number_input("Dots in URL", min_value=0, value=1)
        feat11 = st.number_input("Hyphens Count", min_value=0, value=0)
        feat12 = st.number_input("Underscores Count", min_value=0, value=0)
        feat13 = st.number_input("Slashes Count", min_value=0, value=2)
        feat14 = st.number_input("Question Marks", min_value=0, value=0)
        feat15 = st.number_input("Equal Signs", min_value=0, value=0)
        feat16 = st.number_input("At Symbols (@)", min_value=0, value=0)
    
    # Features 17-24
    with col4:
        feat17 = st.number_input("Ampersands (&)", min_value=0, value=0)
        feat18 = st.number_input("Percent Signs (%)", min_value=0, value=0)
        feat19 = st.number_input("Hash Signs (#)", min_value=0, value=0)
        feat20 = st.number_input("Digits Count", min_value=0, value=2)
        feat21 = st.number_input("Letters Count", min_value=0, value=20)
        feat22 = st.slider("Alexa Rank (normalized)", 0.0, 1.0, 0.5)
        feat23 = st.slider("Domain Trust Score", 0.0, 1.0, 0.8)
        feat24 = st.selectbox("Has SSL Certificate", [0, 1], format_func=lambda x: "Yes" if x else "No")
    
    # Features 25-30
    with col5:
        feat25 = st.number_input("Redirects Count", min_value=0, value=0)
        feat26 = st.slider("Page Load Time (seconds)", 0.0, 10.0, 2.0)
        feat27 = st.selectbox("Has Forms", [0, 1], format_func=lambda x: "Yes" if x else "No")
        feat28 = st.selectbox("Has Hidden Elements", [0, 1], format_func=lambda x: "Yes" if x else "No")
        feat29 = st.slider("External Links Ratio", 0.0, 1.0, 0.3)
        feat30 = st.slider("Image to Text Ratio", 0.0, 1.0, 0.5)
    
    # Prediction button
    if st.button("🔍 Analyze URL", type="primary"):
        if model is not None:
            # Create feature array with exactly 30 features
            features = np.array([[
                url_length, domain_age, subdomain_count, special_chars,
                https_usage, google_index, page_rank, domain_registration_length,
                feat9, feat10, feat11, feat12, feat13, feat14, feat15, feat16,
                feat17, feat18, feat19, feat20, feat21, feat22, feat23, feat24,
                feat25, feat26, feat27, feat28, feat29, feat30
            ]])
            
            # Scale features if scaler is available
            try:
                features_scaled = scaler.transform(features)
            except:
                features_scaled = features
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            # Display results
            st.header("🎯 Prediction Results")
            
            if prediction == 1:  # Assuming 1 = phishing
                st.error(f"⚠️ **PHISHING DETECTED** - Confidence: {prediction_proba[1]:.2%}")
            else:
                st.success(f"✅ **LEGITIMATE URL** - Confidence: {prediction_proba[0]:.2%}")
            
            # Show probability distribution
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Legitimate Probability", f"{prediction_proba[0]:.2%}")
            with col2:
                st.metric("Phishing Probability", f"{prediction_proba[1]:.2%}")

elif input_method == "URL Analysis":
    st.header("URL Analysis")
    st.info("Enter a URL to extract features automatically (requires feature extraction implementation)")
    
    url_input = st.text_input("Enter URL:", placeholder="https://example.com")
    
    if st.button("Analyze URL"):
        if url_input:
            st.warning("Feature extraction from URL not implemented yet. Please use manual input.")
            # Here you would implement URL feature extraction
            # You can use your feature.ipynb logic here
        else:
            st.error("Please enter a URL")

elif input_method == "Batch Prediction":
    st.header("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with features", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
        
        if st.button("Run Batch Prediction"):
            if model is not None:
                try:
                    # Scale features if needed
                    features_scaled = scaler.transform(df)
                    predictions = model.predict(features_scaled)
                    probabilities = model.predict_proba(features_scaled)
                    
                    # Add predictions to dataframe
                    df['Prediction'] = predictions
                    df['Legitimate_Prob'] = probabilities[:, 0]
                    df['Phishing_Prob'] = probabilities[:, 1]
                    df['Status'] = df['Prediction'].map({0: 'Legitimate', 1: 'Phishing'})
                    
                    st.success("Batch prediction completed!")
                    st.dataframe(df[['Prediction', 'Status', 'Legitimate_Prob', 'Phishing_Prob']])
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="phishing_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error in batch prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Built by Group AJ 🎈 | Cybersecurity DLI Project")