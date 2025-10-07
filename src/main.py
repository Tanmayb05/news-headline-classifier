"""
News Headline Classifier using LSTM

Dropping Last FC Layer and use final Hidden State of LSTM to directly predict the category

Author: Tanmay
"""

import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd
import nltk
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Download required nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

torch.backends.cudnn.deterministic = True

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Hyperparameters
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 0.005
BATCH_SIZE = 64
NUM_EPOCHS = 15
DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 4


class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, text, text_length):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(text)
        # ebedded dim: [sentence length, batch size, embedding dim]

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        # Apply fully connected layer for classification
        output = self.fc(hidden)
        return output


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for batch_idx, batch_data in enumerate(data_loader):
            features, text_length = batch_data.TITLE
            targets = batch_data.CATEGORY.to(DEVICE)

            logits = model(features, text_length)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float()/num_examples * 100


def train_model(model, train_loader, valid_loader, optimizer, num_epochs):
    start_time = time.time()
    train_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            features, text_length = batch_data.TITLE
            labels = batch_data.CATEGORY.to(DEVICE)

            # Forward and backward propagation
            logits = model(features, text_length)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            # Update model parameters
            optimizer.step()
            epoch_loss += loss.item()

            # Logging
            if not batch_idx % 1000:
                log_and_print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                      f'Loss: {loss:.4f}')

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        with torch.set_grad_enabled(False):
            train_acc = compute_accuracy(model, train_loader, DEVICE)
            valid_acc = compute_accuracy(model, valid_loader, DEVICE)
            train_accuracies.append(train_acc.item())
            valid_accuracies.append(valid_acc.item())

            log_and_print(f'training accuracy: {train_acc:.2f}%'
                  f'\nvalid accuracy: {valid_acc:.2f}%')

        log_and_print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

    log_and_print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')

    # Return model and training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies
    }
    return model, history


def predict(model, sentence, TEXT, DEVICE):
    model.eval()

    with torch.no_grad():
        # Tokenize using the same method as training
        tokenized = sentence.lower().split()  # Simple whitespace tokenization
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        predict_probas = torch.nn.functional.softmax(model(tensor, length_tensor), dim=1)
        predicted_label_index = torch.argmax(predict_probas)
        predicted_label_proba = torch.max(predict_probas)
        return predicted_label_index.item(), predicted_label_proba.item()


def setup_logging():
    """Setup logging to both console and file"""
    logs_dir = Path('src/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'training_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file


def log_and_print(message, level='info'):
    """Log message and print to console"""
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)


def main():
    # Setup logging
    log_file = setup_logging()
    log_and_print(f"Logging to: {log_file}")

    # Create directories
    log_and_print("=" * 60)
    log_and_print("Setting up directories...")
    log_and_print("=" * 60)
    data_dir = Path('data')
    models_dir = Path('saved_models')
    outputs_dir = Path('outputs')
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    log_and_print(f"✓ Created/verified data directory: {data_dir}")
    log_and_print(f"✓ Created/verified models directory: {models_dir}")
    log_and_print(f"✓ Created/verified outputs directory: {outputs_dir}")

    # Check if dataset exists, download if not
    log_and_print("\n" + "=" * 60)
    log_and_print("Checking for dataset...")
    log_and_print("=" * 60)

    dataset_path = data_dir / 'uci-news-aggregator.csv'
    dataset_gz_path = data_dir / 'uci-news-aggregator.csv.gz'

    if not dataset_path.exists():
        log_and_print("✓ Dataset not found, downloading from source...")
        import urllib.request
        import gzip
        import shutil

        url = "https://github.com/raghavchalapathy/one_class_nn/raw/master/data/uci-news-aggregator.csv.gz"

        try:
            # Download the gzipped file
            log_and_print(f"  └─ Downloading from: {url}")
            urllib.request.urlretrieve(url, dataset_gz_path)
            log_and_print(f"  └─ Downloaded to: {dataset_gz_path}")

            # Unzip the file
            log_and_print(f"  └─ Extracting dataset...")
            with gzip.open(dataset_gz_path, 'rb') as f_in:
                with open(dataset_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log_and_print(f"  └─ Extracted to: {dataset_path}")

            # Remove the gzipped file
            dataset_gz_path.unlink()
            log_and_print(f"  └─ Removed compressed file")
            log_and_print(f"✓ Dataset downloaded and extracted successfully!")

        except Exception as e:
            log_and_print(f"✗ Error downloading dataset: {e}")
            raise
    else:
        log_and_print(f"✓ Dataset found at: {dataset_path}")

    # Load dataset
    log_and_print("\n" + "=" * 60)
    log_and_print("Loading and preprocessing dataset...")
    log_and_print("=" * 60)

    # Always use the original dataset for consistency
    df = pd.read_csv(dataset_path)
    df = df[['TITLE', 'CATEGORY']]

    # Save processed dataset for torchtext to load
    processed_dataset_path = data_dir / 'processed_news.csv'
    df.to_csv(processed_dataset_path, index=False)

    log_and_print(f"✓ Loaded dataset with shape: {df.shape}")
    log_and_print(f"✓ Found categories: {df['CATEGORY'].unique()}")
    category_dist = df['CATEGORY'].value_counts()
    log_and_print(f"✓ Category distribution:")
    for cat, count in category_dist.items():
        log_and_print(f"  - {cat}: {count}")

    # Create visualizations
    log_and_print("\n" + "=" * 60)
    log_and_print("Creating data visualizations...")
    log_and_print("=" * 60)

    # 1. Category distribution
    plt.figure(figsize=(10, 6))
    category_counts = df['CATEGORY'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, palette='viridis', legend=False)
    plt.title('Distribution of News Categories', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    for i, v in enumerate(category_counts.values):
        plt.text(i, v + 1000, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    category_dist_path = outputs_dir / 'category_distribution.png'
    plt.savefig(category_dist_path)
    plt.close()
    log_and_print(f"✓ Saved category distribution: {category_dist_path}")

    # 2. Headline length distribution
    df['TITLE_LENGTH'] = df['TITLE'].str.split().str.len()

    # Create figure with subplots for better clarity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Headline Lengths by Category', fontsize=16, fontweight='bold')

    categories = sorted(df['CATEGORY'].unique())
    colors = {'e': '#1f77b4', 'b': '#ff7f0e', 't': '#2ca02c', 'm': '#d62728'}

    # Calculate appropriate bins based on actual data range
    min_len = df['TITLE_LENGTH'].min()
    max_len = df['TITLE_LENGTH'].max()
    bins = range(int(min_len), int(max_len) + 2, 1)  # 1-word bins

    for idx, category in enumerate(categories):
        ax = axes[idx // 2, idx % 2]
        category_data = df[df['CATEGORY'] == category]['TITLE_LENGTH']

        ax.hist(category_data, bins=bins, alpha=0.7, color=colors.get(category, 'gray'), edgecolor='black')
        ax.set_title(f'Category {category.upper()} (n={len(category_data):,})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Words', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_len = category_data.mean()
        median_len = category_data.median()
        ax.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
        ax.axvline(median_len, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
        ax.legend(fontsize=8)
        ax.set_xlim(min_len - 1, min(max_len + 1, 50))  # Cap at 50 for better visualization

    plt.tight_layout()
    length_dist_path = outputs_dir / 'headline_length_distribution.png'
    plt.savefig(length_dist_path)
    plt.close()
    log_and_print(f"✓ Saved headline length distribution: {length_dist_path}")

    # 3. Category pie chart
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('pastel')[0:len(category_counts)]
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Category Distribution (Percentage)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    pie_chart_path = outputs_dir / 'category_pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()
    log_and_print(f"✓ Saved category pie chart: {pie_chart_path}")

    # Define feature and label processing
    log_and_print("\n" + "=" * 60)
    log_and_print("Initializing text and label processors...")
    log_and_print("=" * 60)

    # Define simple tokenizer function
    def simple_tokenizer(text):
        return text.lower().split()

    TEXT = torchtext.data.Field(
        tokenize=simple_tokenizer,  # Use simple whitespace tokenizer
        include_lengths=True
    )

    LABEL = torchtext.data.LabelField(dtype=torch.long)
    log_and_print("✓ Text field configured with simple tokenizer")
    log_and_print("✓ Label field configured")

    # Create dataset and split
    log_and_print("\n" + "=" * 60)
    log_and_print("Creating dataset and splitting into train/validation/test sets...")
    log_and_print("=" * 60)

    fields = [('TITLE', TEXT), ('CATEGORY', LABEL)]
    dataset = torchtext.data.TabularDataset(
        path=str(processed_dataset_path),  # Use the processed dataset with only TITLE and CATEGORY
        format='csv',
        skip_header=True,
        fields=fields
    )
    log_and_print(f"✓ Created TabularDataset with {len(dataset)} examples")

    # Save test data BEFORE splitting (from original dataset)
    log_and_print("\n" + "=" * 60)
    log_and_print("Saving test dataset for app usage...")
    log_and_print("=" * 60)

    # Use same random seed to get consistent split
    total_len = len(df)
    test_size = int(total_len * 0.2)

    # Shuffle with same seed
    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df_export = df_shuffled.iloc[-test_size:]

    test_csv_path = data_dir / 'test.csv'
    test_df_export.to_csv(test_csv_path, index=False)
    log_and_print(f'✓ Saved test data to: {test_csv_path} ({len(test_df_export)} samples)')

    # Split dataset (with fixed random seed for reproducibility)
    train_data, test_data = dataset.split(
        split_ratio=[0.8, 0.2],
        random_state=random.seed(RANDOM_SEED)
    )

    train_data, valid_data = train_data.split(
        split_ratio=[0.85, 0.15],
        random_state=random.seed(RANDOM_SEED)
    )

    log_and_print(f'✓ Training samples: {len(train_data)} (68%)')
    log_and_print(f'✓ Validation samples: {len(valid_data)} (12%)')
    log_and_print(f'✓ Test samples: {len(test_data)} (20%)')

    # Build vocabulary
    log_and_print("\n" + "=" * 60)
    log_and_print("Building vocabulary from training data...")
    log_and_print("=" * 60)
    TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
    LABEL.build_vocab(train_data)

    log_and_print(f'✓ Vocabulary size: {len(TEXT.vocab)} (max: {VOCABULARY_SIZE})')
    log_and_print(f'✓ Number of classes: {len(LABEL.vocab)}')
    class_mapping = dict(LABEL.vocab.stoi)
    log_and_print(f'✓ Class mapping: {", ".join([f"{k}:{v}" for k, v in class_mapping.items()])}')

    # Create data loaders
    log_and_print("\n" + "=" * 60)
    log_and_print("Creating data loaders with BucketIterator...")
    log_and_print("=" * 60)
    train_loader, valid_loader, test_loader = \
        torchtext.data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            sort_key=lambda x: len(x.TITLE),
            device=DEVICE
        )
    log_and_print(f'✓ Batch size: {BATCH_SIZE}')
    log_and_print(f'✓ Train batches: {len(train_loader)}')
    log_and_print(f'✓ Validation batches: {len(valid_loader)}')
    log_and_print(f'✓ Test batches: {len(test_loader)}')

    # Check if model already exists
    model_path = models_dir / 'lstm_headline_classifier.pt'
    model_exists = model_path.exists()

    # Initialize model
    log_and_print("\n" + "=" * 60)
    log_and_print("Initializing LSTM model...")
    log_and_print("=" * 60)
    torch.manual_seed(RANDOM_SEED)
    model = RNN(
        input_dim=len(TEXT.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=NUM_CLASSES
    )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_and_print(f'✓ Model initialized and moved to device: {DEVICE}')
    log_and_print(f'✓ Total parameters: {total_params:,}')
    log_and_print(f'✓ Trainable parameters: {trainable_params:,}')
    log_and_print(f'✓ Optimizer: Adam (lr={LEARNING_RATE})')
    log_and_print(f'✓ Model architecture:')
    log_and_print(f'  - Embedding dim: {EMBEDDING_DIM}')
    log_and_print(f'  - Hidden dim: {HIDDEN_DIM}')
    log_and_print(f'  - Output classes: {NUM_CLASSES}')

    # Check if saved model exists
    if model_exists:
        log_and_print("\n" + "=" * 60)
        log_and_print(f"Found existing trained model at: {model_path}")
        log_and_print("=" * 60)
        log_and_print("Loading existing model (idempotent execution)...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_and_print(f"✓ Loaded model from checkpoint")
        log_and_print(f"✓ Saved test accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")

        # Evaluate loaded model
        log_and_print("\n" + "=" * 60)
        log_and_print("Evaluating loaded model on test set...")
        log_and_print("=" * 60)
        test_acc = compute_accuracy(model, test_loader, DEVICE)
        log_and_print(f'✓ Current Test Accuracy: {test_acc:.2f}%')

        # Load training history if available
        history = checkpoint.get('history', None)
    else:
        # Train model (no existing model found)
        log_and_print("\n" + "=" * 60)
        log_and_print("No existing model found. Starting training...")
        log_and_print("=" * 60)
        model, history = train_model(model, train_loader, valid_loader, optimizer, NUM_EPOCHS)

        # Evaluate on test set
        log_and_print("\n" + "=" * 60)
        log_and_print("Evaluating model on test set...")
        log_and_print("=" * 60)
        test_acc = compute_accuracy(model, test_loader, DEVICE)
        log_and_print(f'✓ Final Test Accuracy: {test_acc:.2f}%')

        # Save model
        log_and_print("\n" + "=" * 60)
        log_and_print("Saving trained model...")
        log_and_print("=" * 60)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': len(TEXT.vocab),
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_classes': NUM_CLASSES,
            'test_accuracy': test_acc.item(),
            'history': history,
            'vocab_stoi': dict(TEXT.vocab.stoi),  # Save word to index mapping
            'vocab_itos': TEXT.vocab.itos,  # Save index to word mapping
            'label_stoi': dict(LABEL.vocab.stoi),  # Save label to index mapping
            'label_itos': LABEL.vocab.itos,  # Save index to label mapping
        }, model_path)
        log_and_print(f'✓ Model saved to: {model_path}')
        log_and_print(f'✓ Model checkpoint includes: state_dict, optimizer, hyperparameters, and test accuracy')

        # Create training visualizations
        if history:
            log_and_print("\n" + "=" * 60)
            log_and_print("Creating training visualizations...")
            log_and_print("=" * 60)

            epochs = range(1, NUM_EPOCHS + 1)

            # Create comprehensive training plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Training Progress and Performance Metrics', fontsize=18, fontweight='bold', y=0.995)

            # 1. Training Loss
            axes[0, 0].plot(epochs, history['train_losses'], 'b-', linewidth=2.5, marker='o', markersize=4)
            axes[0, 0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(1, NUM_EPOCHS)

            # 2. Training vs Validation Accuracy
            axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy',
                           linewidth=2.5, marker='o', markersize=4)
            axes[0, 1].plot(epochs, history['valid_accuracies'], 'r-', label='Validation Accuracy',
                           linewidth=2.5, marker='s', markersize=4)
            axes[0, 1].axhline(y=test_acc.item(), color='g', linestyle='--', linewidth=2,
                              label=f'Test Accuracy: {test_acc:.2f}%')
            axes[0, 1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
            axes[0, 1].legend(loc='lower right', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(1, NUM_EPOCHS)

            # 3. Accuracy Gap (Overfitting indicator)
            accuracy_gap = [train - valid for train, valid in zip(history['train_accuracies'],
                                                                   history['valid_accuracies'])]
            axes[1, 0].plot(epochs, accuracy_gap, 'purple', linewidth=2.5, marker='^', markersize=4)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[1, 0].fill_between(epochs, 0, accuracy_gap, alpha=0.3, color='purple')
            axes[1, 0].set_title('Train-Validation Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Accuracy Gap (%)', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(1, NUM_EPOCHS)

            # 4. Performance Summary
            axes[1, 1].axis('off')
            summary_text = f"""
            TRAINING SUMMARY
            {'='*40}

            Total Epochs: {NUM_EPOCHS}

            Final Training Accuracy: {history['train_accuracies'][-1]:.2f}%
            Final Validation Accuracy: {history['valid_accuracies'][-1]:.2f}%
            Test Accuracy: {test_acc:.2f}%

            Final Training Loss: {history['train_losses'][-1]:.4f}

            Best Validation Accuracy: {max(history['valid_accuracies']):.2f}%
            (Epoch {history['valid_accuracies'].index(max(history['valid_accuracies'])) + 1})

            Overfitting Gap: {accuracy_gap[-1]:.2f}%

            Model Parameters: {total_params:,}
            Vocabulary Size: {len(TEXT.vocab):,}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                          fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            training_curves_path = outputs_dir / 'training_metrics.png'
            plt.savefig(training_curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            log_and_print(f"✓ Saved training metrics: {training_curves_path}")

    # Make predictions
    log_and_print("\n" + "=" * 60)
    log_and_print("Making sample predictions on custom sentences...")
    log_and_print("=" * 60)
    class_mapping = LABEL.vocab.stoi
    inverse_class_mapping = {v: k for k, v in class_mapping.items()}

    test_sentences = [
        "Stock Falls",
        "Today is a good day"
    ]

    for sentence in test_sentences:
        predicted_label_index, predicted_label_proba = predict(model, sentence, TEXT, DEVICE)
        predicted_label = inverse_class_mapping[predicted_label_index]
        log_and_print(f'✓ Sentence: "{sentence}"')
        log_and_print(f'  └─ Predicted label: {predicted_label} (index: {predicted_label_index})')
        log_and_print(f'  └─ Confidence: {predicted_label_proba:.6f}\n')

    log_and_print("\n" + "=" * 60)
    log_and_print("Training completed successfully!")
    log_and_print(f"Full log saved to: {log_file}")
    log_and_print("=" * 60)


if __name__ == "__main__":
    main()
