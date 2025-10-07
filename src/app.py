# src/app.py
import streamlit as st
import torch
import torch.nn.functional as F
import spacy
import os
import pandas as pd
import random

# Import Model Class
from model import RNN

# --- Page Configuration ---
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)

# --- Emojis for Categories ---
CATEGORY_EMOJIS = {
    "E": "🎭 Entertainment",
    "B": "💼 Business",
    "T": "🔬 Sci/Tech",
    "M": "❤️ Health"
}

# --- Configuration ---
# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.pt')
FIELDS_PATH = os.path.join(MODEL_DIR, 'fields.pt')

# --- Load Model and Fields ---
@st.cache_resource
def load_model():
    """Load the model and fields once and cache them."""
    nlp = spacy.blank("en")

    # Load the vocabulary and model artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FIELDS_PATH):
        st.error("Model or fields file not found. Please run train.py first.")
        st.stop()

    fields_loaded = torch.load(FIELDS_PATH, weights_only=False)
    TEXT = fields_loaded['TEXT']
    LABEL = fields_loaded['LABEL']

    model = RNN(
        input_dim=len(TEXT.vocab),
        embedding_dim=128,
        hidden_dim=256,
        output_dim=len(LABEL.vocab)
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, TEXT, LABEL, nlp

# Load model
model, TEXT, LABEL, nlp = load_model()

@st.cache_data
def load_sample_headlines():
    """Load sample headlines from the merged headlines file."""
    try:
        with open('data/all_headlines.txt', 'r') as f:
            headlines = [line.strip() for line in f if line.strip()]
        return headlines
    except Exception:
        # Fallback headlines if data file is not available
        return [
            "Apple announces new iPhone at annual event",
            "Stock market hits record high amid economic growth",
            "Scientists discover new species in Amazon rainforest",
            "New study reveals benefits of Mediterranean diet",
            "Hollywood star wins Academy Award for best actor",
            "Tech giant releases revolutionary AI software",
            "Major breakthrough in cancer treatment research",
            "Championship team wins in dramatic final match",
            "New smartphone features advanced camera technology",
            "Researchers develop vaccine for rare disease"
        ]

def predict_category(sentence):
    """Predicts the category of a single sentence."""
    with torch.no_grad():
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE).unsqueeze(1)
        length_tensor = torch.LongTensor(length)

        prediction = model(tensor, length_tensor)
        probabilities = F.softmax(prediction, dim=1)

        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_prob = torch.max(probabilities).item()

        return LABEL.vocab.itos[predicted_class_index], predicted_class_prob

# Load sample headlines
sample_headlines = load_sample_headlines()

# --- UI Elements ---
st.title("📰 News Category Classifier")
st.markdown("Enter a news headline below to classify it into one of the four categories.")

# Button to generate random headline
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🎲 Random", use_container_width=True):
        st.session_state.headline = random.choice(sample_headlines)

# Text input box
headline_text = st.text_input(
    "Enter Headline:",
    placeholder="e.g., 'Apple announces new iPhone at annual event'",
    value=st.session_state.get('headline', ''),
    key='headline_input'
)

# Predict button
if st.button("Classify Headline 🚀"):
    if headline_text:
        with st.spinner("🧠 Analyzing..."):
            try:
                # --- Prediction ---
                category, probability = predict_category(headline_text)

                # Get the emoji and full name for the category
                display_category = CATEGORY_EMOJIS.get(category.upper(), f"❓ {category}")

                st.success(f"**Prediction:** {display_category}")
                st.metric(label="Confidence", value=f"{probability:.2%}")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a headline to classify.")

# --- Footer ---
st.markdown("---")
st.markdown("Powered by a Bidirectional LSTM model built with PyTorch.")
