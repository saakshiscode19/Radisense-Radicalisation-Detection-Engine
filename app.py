import streamlit as st
import torch
import torch.nn.functional as F
import re
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer

# --- Page Configuration ---
st.set_page_config(
    page_title="Radicalisation Detector",
    page_icon="🔍",
    layout="centered"
)

# --- Constants ---
# Detailed labels for the UI and Plotly Graph
LABEL_NAMES = [
    "🟢 NEUTRAL (Objective News/Facts)",
    "🟡 NON-RADICAL (Standard Social Interaction)",
    "🔴 RADICAL (Extremist Propaganda / Ideology)"
]
LABEL_COLORS = ["#2ecc71", "#f1c40f", "#e74c3c"]  # Green, Yellow, Red

# Short labels specifically for LIME to prevent text overlap in its visualization
LIME_SHORT_NAMES = ["Neutral", "Non-Radical", "Radical"] 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model & Tokenizer ---
@st.cache_resource
def load_model():
    model_path = './radical_model'  # Points to your saved folder
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Utility Functions ---
def clean_text(text):
    """Preprocesses the text exactly as done during your training phase."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict(text):
    """Runs inference on a single text string."""
    cleaned = clean_text(text)
    enc = tokenizer(
        cleaned, 
        return_tensors="pt", 
        truncation=True,
        padding=True, 
        max_length=128
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=1)[0]
    
    pred_idx = torch.argmax(probs).item()
    conf = probs[pred_idx].item() * 100
    
    return pred_idx, conf, probs.cpu().numpy()

def lime_predict_proba(texts):
    """LIME requires a function that takes a list of strings and returns a numpy array of probabilities."""
    enc = tokenizer(
        list(texts), 
        truncation=True, 
        padding=True, 
        max_length=128, 
        return_tensors='pt'
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = model(**enc).logits
    return F.softmax(logits, dim=1).cpu().numpy()

# --- Streamlit UI ---
st.title("🔍 RadiSense : Radicalisation Detection Engine")
st.markdown("""
Enter a text snippet below to analyze it. The model will classify the text and provide a **LIME (Local Interpretable Model-agnostic Explanations)** breakdown showing which words drove the decision.
""")

# Input Area
user_input = st.text_area("Enter text for analysis:", height=150, placeholder="Type or paste text here...")

# Analyze Button
if st.button("Analyze Text", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text with RoBERTa..."):
            # Run inference
            pred_idx, conf, probs_array = predict(user_input)
            
            # Formatting results
            predicted_class = LABEL_NAMES[pred_idx]
            color = LABEL_COLORS[pred_idx]
            
            # Display Prediction
            st.markdown("### Analysis Result")
            st.markdown(
                f"<h3 style='color: {color}; margin-top: 0;'>{predicted_class}<br><small style='color: gray;'>Confidence: {conf:.2f}%</small></h3>", 
                unsafe_allow_html=True
            )
            
            st.divider()
            
            # Display Probability Graph
            st.markdown("### Probability Distribution")
            
            df_probs = pd.DataFrame({
                "Class": LABEL_NAMES,
                "Probability": probs_array,
            })
            
            # Create Plotly Bar Chart
            fig = px.bar(
                df_probs, 
                x="Probability", 
                y="Class",
                text="Probability",
                orientation='h', 
                color="Class",
                color_discrete_map={
                    "🟢 NEUTRAL (Objective News/Facts)": "#2ecc71", 
                    "🟡 NON-RADICAL (Standard Social Interaction)": "#f1c40f", 
                    "🔴 RADICAL (Extremist Propaganda / Ideology)": "#e74c3c"
                }
            )
            
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(
                xaxis=dict(range=[0, 1.1], tickformat=".0%"),
                showlegend=False,
                margin=dict(l=20, r=40, t=30, b=20),
                height=350,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        st.divider()
        
        # --- LIME Explainability Section ---
        st.markdown("### Model Explainability (LIME)")
        with st.spinner("Generating LIME Explanation (this takes a moment)..."):
            
            # IMPORTANT: Using LIME_SHORT_NAMES here to prevent the UI text overlap!
            explainer = LimeTextExplainer(class_names=LIME_SHORT_NAMES)
            
            # Generate the explanation specifically for the predicted class
            exp = explainer.explain_instance(
                user_input, 
                lime_predict_proba, 
                num_features=10, 
                num_samples=300, 
                labels=[pred_idx]
            )
            
            # Render LIME's interactive HTML output in Streamlit
            html_output = exp.as_html(labels=[pred_idx])
            
            # Increased height so LIME has plenty of room to render properly without squishing
            components.html(html_output, height=550, scrolling=True)

        with st.expander("Show preprocessed text"):
            st.write(f"_{clean_text(user_input)}_")