# src/app.py
import streamlit as st
import torch
import torch.nn.functional as F
import os
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the RNN model class
class RNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_length):
        embedded = self.embedding(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed)
        hidden.squeeze_(0)
        output = self.fc(hidden)
        return output

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
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_headline_classifier.pt')

# Category mapping
CATEGORY_MAPPING = {
    0: 'E',
    1: 'B',
    2: 'T',
    3: 'M'
}

# --- Load Model ---
@st.cache_resource
def load_model():
    """Load the model checkpoint once and cache it."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run src/main.py first to train the model.")
        st.stop()

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Initialize model with checkpoint parameters
    model = RNN(
        input_dim=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['num_classes']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

# Load model
model, checkpoint = load_model()

# Simple tokenizer (matching main.py's basic_english tokenizer)
def tokenize(text):
    """Simple tokenization matching the training script."""
    return text.lower().split()

@st.cache_data
def load_sample_headlines():
    """Load sample headlines from test.csv file."""
    try:
        # Load from test.csv
        test_df = pd.read_csv('data/test.csv')
        if 'TITLE' in test_df.columns:
            headlines = test_df['TITLE'].tolist()
            return headlines
        else:
            raise ValueError("TITLE column not found in test.csv")
    except Exception as e:
        # Fallback headlines if data file is not available
        st.warning(f"Could not load test.csv: {e}. Using fallback headlines.")
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
        # Tokenize (must match training: lowercase + split)
        tokenized = sentence.lower().split()

        # Get vocab from checkpoint
        vocab_stoi = checkpoint.get('vocab_stoi', {})
        label_itos = checkpoint.get('label_itos', ['e', 'b', 't', 'm'])

        # Debug: Check if vocab is loaded
        if not vocab_stoi:
            raise ValueError("Vocabulary not found in checkpoint. Please retrain the model with updated main.py")

        # Convert words to indices using saved vocabulary
        # Use <unk> token (index 0) for unknown words
        indexed = [vocab_stoi.get(word, 0) for word in tokenized]

        if len(indexed) == 0:
            indexed = [0]  # Handle empty input

        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE).unsqueeze(1)
        length_tensor = torch.LongTensor(length)

        prediction = model(tensor, length_tensor)
        probabilities = F.softmax(prediction, dim=1)

        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_prob = torch.max(probabilities).item()

        # Use the label mapping from checkpoint
        predicted_category = label_itos[predicted_class_index].upper() if predicted_class_index < len(label_itos) else 'Unknown'
        return predicted_category, predicted_class_prob

# Load sample headlines
sample_headlines = load_sample_headlines()

# --- UI Elements ---
st.title("📰 News Category Classifier")

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs(["ℹ️ Info", "🎯 Classify", "📊 Visualizations"])

# --- Info Tab ---
with tab1:
    st.header("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", f"{checkpoint.get('test_accuracy', 0):.2f}%")
        st.metric("Vocabulary Size", f"{checkpoint.get('vocab_size', 0):,}")

    with col2:
        st.metric("Embedding Dimension", checkpoint.get('embedding_dim', 0))
        st.metric("Hidden Dimension", checkpoint.get('hidden_dim', 0))

    with col3:
        st.metric("Number of Classes", checkpoint.get('num_classes', 0))
        st.metric("Device", str(DEVICE).upper())

    st.markdown("---")

    st.subheader("Categories")
    for key, value in CATEGORY_EMOJIS.items():
        st.write(f"**{key}**: {value}")

    st.markdown("---")

    st.subheader("About the Model")
    st.write("""
    This classifier uses a **Bidirectional LSTM** architecture built with PyTorch to categorize news headlines
    into four categories: Entertainment, Business, Sci/Tech, and Health.

    **Key Features:**
    - Embedding Layer: Converts text tokens to dense vectors
    - Bidirectional LSTM: Processes text in both directions for better context understanding
    - Fully Connected Layer: Final classification layer
    - Dropout Regularization: Prevents overfitting

    **Note:** Low confidence doesn't necessarily mean a wrong prediction. The model achieves high accuracy on the test set.
    """)

# --- Classify Tab ---
with tab2:
    st.markdown("Enter a news headline below to classify it into one of the four categories.")
    st.info("💡 **Tip:** Click the 'Random' button to generate a random headline. Wait for some time to load to generate random lines.")

    # Initialize session state for headline
    if 'random_headline' not in st.session_state:
        st.session_state.random_headline = ''

    # Button to generate random headline
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.random_headline = str(random.choice(sample_headlines))

    # Text input box
    headline_text = st.text_input(
        "Enter Headline:",
        placeholder="e.g., 'Apple announces new iPhone at annual event'",
        value=st.session_state.random_headline
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

# --- Visualizations Tab ---
with tab3:
    st.header("Data & Training Visualizations")

    # Load dataset for visualizations
    try:
        # Try loading processed_news.csv first
        if os.path.exists('data/processed_news.csv'):
            df = pd.read_csv('data/processed_news.csv')
        else:
            st.warning("Dataset not found. Please run `python src/main.py` to generate the dataset.")
            st.stop()

        # Category mapping for display
        category_map = {
            'b': '💼 Business',
            'e': '🎭 Entertainment',
            't': '🔬 Sci/Tech',
            'm': '❤️ Health'
        }
        df['Category_Name'] = df['CATEGORY'].str.lower().map(category_map)

        # 1. Category Distribution Bar Chart
        st.subheader("📊 Category Distribution")
        category_counts = df['Category_Name'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']

        fig_bar = px.bar(
            category_counts,
            x='Category',
            y='Count',
            title='Distribution of News Categories',
            color='Category',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_traces(
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
            marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)')),
            texttemplate='%{y}',
            textposition='outside',
            textfont=dict(size=14, color='black')
        )
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Category",
            yaxis_title="Number of Headlines",
            xaxis=dict(tickfont=dict(color='black', size=12)),
            yaxis=dict(tickfont=dict(color='black', size=12)),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        # 2. Category Distribution Pie Chart
        st.subheader("🥧 Category Distribution (Pie Chart)")
        fig_pie = px.pie(
            category_counts,
            values='Count',
            names='Category',
            title='Category Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=14, color='black'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            marker=dict(line=dict(color='white', width=2)),
            pull=[0.05, 0.05, 0.05, 0.05]  # Slight pull effect for all slices
        )
        fig_pie.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            ),
            font=dict(color='black')
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        # 3. Headline Length Distribution by Category
        st.subheader("📏 Headline Length Distribution by Category")
        df['headline_length'] = df['TITLE'].str.split().str.len()

        # Create 2x2 subplots for each category
        categories = sorted(df['CATEGORY'].str.lower().unique())
        category_colors = {'e': '#1f77b4', 'b': '#ff7f0e', 't': '#2ca02c', 'm': '#d62728'}
        category_names = {
            'e': '🎭 Entertainment',
            'b': '💼 Business',
            't': '🔬 Sci/Tech',
            'm': '❤️ Health'
        }

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{category_names.get(cat, cat.upper())} (n={len(df[df['CATEGORY'].str.lower() == cat]):,})"
                          for cat in categories]
        )

        min_len = df['headline_length'].min()
        max_len = df['headline_length'].max()

        for idx, category in enumerate(categories):
            row = idx // 2 + 1
            col = idx % 2 + 1

            category_data = df[df['CATEGORY'].str.lower() == category]['headline_length']

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=category_data,
                    nbinsx=int(max_len - min_len + 1),
                    marker_color=category_colors.get(category, 'gray'),
                    marker_line=dict(color=category_colors.get(category, 'gray'), width=1.5),
                    opacity=0.7,
                    name=category.upper(),
                    showlegend=False,
                    hovertemplate='Words: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=row, col=col
            )

            # Calculate statistics
            mean_len = category_data.mean()
            median_len = category_data.median()

            # Add mean line
            fig.add_vline(
                x=mean_len,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Mean: {mean_len:.1f}",
                annotation_position="top right",
                annotation_yshift=10,
                row=row, col=col
            )

            # Add median line
            fig.add_vline(
                x=median_len,
                line_dash="dash",
                line_color="blue",
                line_width=2,
                annotation_text=f"Median: {median_len:.1f}",
                annotation_position="bottom right",
                annotation_yshift=-10,
                row=row, col=col
            )

            # Update axes
            fig.update_xaxes(title_text="Number of Words", row=row, col=col, range=[min_len - 1, min(max_len + 1, 50)])
            fig.update_yaxes(title_text="Frequency", row=row, col=col)

        fig.update_layout(
            height=800,
            title_text="Distribution of Headline Lengths by Category",
            showlegend=False,
            title_y=0.98,
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Arial"
            ),
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 4. Training Metrics (if available in checkpoint)
        if 'history' in checkpoint and checkpoint['history']:
            history = checkpoint['history']

            # Check if history has the required keys
            if 'train_losses' in history and 'train_accuracies' in history and 'valid_accuracies' in history:
                st.subheader("📈 Training Metrics")

                # Create subplots for training metrics
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Training Loss', 'Training vs Validation Accuracy', 'Training Accuracy', 'Validation Accuracy')
                )

                epochs = list(range(1, len(history['train_losses']) + 1))

                # Training Loss
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=history['train_losses'],
                        mode='lines+markers',
                        name='Train Loss',
                        line=dict(color='#EF553B', width=3),
                        marker=dict(size=8, line=dict(width=2, color='white')),
                        hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Training vs Validation Accuracy (combined)
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=history['train_accuracies'],
                        mode='lines+markers',
                        name='Train Acc',
                        line=dict(color='#636EFA', width=3),
                        marker=dict(size=8, line=dict(width=2, color='white')),
                        hovertemplate='Epoch: %{x}<br>Train Acc: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=history['valid_accuracies'],
                        mode='lines+markers',
                        name='Val Acc',
                        line=dict(color='#EF553B', width=3),
                        marker=dict(size=8, symbol='square', line=dict(width=2, color='white')),
                        hovertemplate='Epoch: %{x}<br>Val Acc: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=2
                )

                # Add test accuracy line
                test_acc = checkpoint.get('test_accuracy', 0)
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=[test_acc] * len(epochs),
                        mode='lines',
                        name='Test Acc',
                        line=dict(color='#00CC96', width=2.5, dash='dash'),
                        hovertemplate='Test Accuracy: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=2
                )

                # Training Accuracy
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=history['train_accuracies'],
                        mode='lines+markers',
                        name='Train Acc',
                        line=dict(color='#636EFA', width=3),
                        marker=dict(size=8, line=dict(width=2, color='white')),
                        hovertemplate='Epoch: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )

                # Validation Accuracy
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=history['valid_accuracies'],
                        mode='lines+markers',
                        name='Val Acc',
                        line=dict(color='#AB63FA', width=3),
                        marker=dict(size=8, line=dict(width=2, color='white')),
                        hovertemplate='Epoch: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=2
                )

                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                fig.update_xaxes(title_text="Epoch", row=2, col=1)
                fig.update_xaxes(title_text="Epoch", row=2, col=2)

                fig.update_yaxes(title_text="Loss", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
                fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
                fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)

                fig.update_layout(
                    height=800,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=14,
                        font_family="Arial"
                    ),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # Training Summary
                st.markdown("### 📋 Training Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Training Progress:**")
                    st.write(f"- Total Epochs: {len(history['train_losses'])}")
                    st.write(f"- Final Train Loss: {history['train_losses'][-1]:.4f}")
                    st.write(f"- Final Train Accuracy: {history['train_accuracies'][-1]:.2f}%")

                with col2:
                    st.write("**Validation Performance:**")
                    st.write(f"- Final Val Accuracy: {history['valid_accuracies'][-1]:.2f}%")
                    st.write(f"- Best Val Accuracy: {max(history['valid_accuracies']):.2f}%")

                with col3:
                    st.write("**Test Performance:**")
                    st.write(f"- Test Accuracy: {test_acc:.2f}%")

    except Exception as e:
        st.error(f"Error loading visualizations: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Powered by a Bidirectional LSTM model built with PyTorch.")
