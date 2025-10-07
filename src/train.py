# -*- coding: utf-8 -*-
"""
This script trains the LSTM model for text classification.
It checks for a pre-trained model to skip training on subsequent runs.
"""

import torch
import pandas as pd
import time
import random
import spacy
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

# Import legacy torchtext modules
from torchtext import data

# Import the model class from the new model.py
from model import RNN

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Main execution block ---
# This ensures the training code only runs when the script is executed directly
if __name__ == "__main__":

    # --- Configuration ---
    RANDOM_SEED = 123
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    VOCABULARY_SIZE = 20000
    LEARNING_RATE = 0.005
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    logging.info(f"Using device: {DEVICE}")

    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 4

    # --- Path Definitions ---
    DATA_DIR = '../data'
    MODEL_DIR = 'saved_models'
    OUTPUT_DIR = '../outputs'
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, 'lstm_model.pt')
    fields_path = os.path.join(MODEL_DIR, 'fields.pt')

    if os.path.exists(model_path) and os.path.exists(fields_path):
        logging.info("Found pre-trained model and fields. Skipping training.")
        
        fields_loaded = torch.load(fields_path, weights_only=False)
        TEXT = fields_loaded['TEXT']
        LABEL = fields_loaded['LABEL']
        
        model = RNN(
            input_dim=len(TEXT.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(model_path))
        
        fields_list = [('TITLE', TEXT), ('CATEGORY', LABEL)]
        test_data = data.TabularDataset(
            path=os.path.join(DATA_DIR, 'test.csv'),
            format='csv',
            fields=fields_list,
            skip_header=True
        )
        test_iterator = data.BucketIterator(
            test_data, batch_size=BATCH_SIZE,
            sort_within_batch=True, sort_key=lambda x: len(x.TITLE), device=DEVICE
        )

    else:
        logging.info("No pre-trained model found. Starting full training process.")
        
        raw_data_path = os.path.join(DATA_DIR, 'uci-news-aggregator.csv')
        if not os.path.exists(raw_data_path):
            logging.info("Downloading data...")
            os.system(f"wget -q https://github.com/raghavchalapathy/one_class_nn/raw/master/data/uci-news-aggregator.csv.gz -P {DATA_DIR}")
            os.system(f"gunzip -f {os.path.join(DATA_DIR, 'uci-news-aggregator.csv.gz')}")
            logging.info("Data downloaded and unzipped.")

        df = pd.read_csv(raw_data_path)
        df = df[['TITLE', 'CATEGORY']]
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['CATEGORY'])
        train_df, valid_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_SEED, stratify=train_df['CATEGORY'])
        
        train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
        valid_df.to_csv(os.path.join(DATA_DIR, 'valid.csv'), index=False)
        test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
        
        logging.info(f'Num Train: {len(train_df)}, Num Validation: {len(valid_df)}, Num Test: {len(test_df)}')

        try:
            spacy.load('en_core_web_sm')
        except IOError:
            logging.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
            os.system("python -m spacy download en_core_web_sm")
        TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
        LABEL = data.LabelField(dtype=torch.long)
        fields = [('TITLE', TEXT), ('CATEGORY', LABEL)]

        train_data, valid_data, test_data = data.TabularDataset.splits(
            path=DATA_DIR, train='train.csv', validation='valid.csv', test='test.csv',
            format='csv', fields=fields, skip_header=True
        )

        TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
        LABEL.build_vocab(train_data)
        logging.info(f"Vocabulary size: {len(TEXT.vocab)}")

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), batch_size=BATCH_SIZE,
            sort_within_batch=True, sort_key=lambda x: len(x.TITLE), device=DEVICE
        )

        model = RNN(
            input_dim=len(TEXT.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)

        logging.info("Starting model training...")
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_start_time = time.time()
            for i, batch in enumerate(train_iterator):
                features, text_lengths = batch.TITLE
                labels = batch.CATEGORY
                logits = model(features, text_lengths)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not i % 200:
                    logging.info(f'Epoch: {epoch+1:02d}/{NUM_EPOCHS:02d} | Batch {i:04d}/{len(train_iterator):04d} | Loss: {loss:.4f}')
            
            def compute_accuracy(model, iterator, device):
                model.eval()
                correct_pred, num_examples = 0, 0
                with torch.no_grad():
                    for batch in iterator:
                        features, text_lengths = batch.TITLE
                        targets = batch.CATEGORY
                        logits = model(features, text_lengths)
                        _, predicted_labels = torch.max(logits, 1)
                        num_examples += targets.size(0)
                        correct_pred += (predicted_labels == targets).sum()
                return correct_pred.float() / num_examples * 100

            with torch.set_grad_enabled(False):
                valid_acc = compute_accuracy(model, valid_iterator, DEVICE)
                logging.info(f'--- Epoch {epoch+1:02d} Summary ---')
                logging.info(f'Validation Accuracy: {valid_acc:.2f}%')
                logging.info(f'Epoch Time: {(time.time() - epoch_start_time):.2f}s')
        
        logging.info(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')

        logging.info("Saving model and fields...")
        torch.save(model.state_dict(), model_path)
        torch.save({'TEXT': TEXT, 'LABEL': LABEL}, fields_path)
        logging.info(f"Model and fields saved to {MODEL_DIR}")

    # --- Model Evaluation ---
    logging.info("Starting model evaluation on the test set...")

    def evaluate_model(model, iterator, device):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in iterator:
                features, text_lengths = batch.TITLE
                targets = batch.CATEGORY
                logits = model(features, text_lengths)
                _, predicted_labels = torch.max(logits, 1)
                all_preds.extend(predicted_labels.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        class_names = [LABEL.vocab.itos[i] for i in range(len(LABEL.vocab))]
        report = classification_report(all_targets, all_preds, target_names=class_names)
        logging.info("\nClassification Report:\n" + report)
        
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        logging.info(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")
        plt.show()

    def compute_test_accuracy(model, iterator, device):
        model.eval()
        correct_pred, num_examples = 0, 0
        with torch.no_grad():
            for batch in iterator:
                features, text_lengths = batch.TITLE
                targets = batch.CATEGORY
                logits = model(features, text_lengths)
                _, predicted_labels = torch.max(logits, 1)
                num_examples += targets.size(0)
                correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples * 100

    test_acc = compute_test_accuracy(model, test_iterator, DEVICE)
    logging.info(f'Test Accuracy: {test_acc:.2f}%')
    evaluate_model(model, test_iterator, DEVICE)
    logging.info("Model evaluation complete.")

    # --- Prediction on New Sentences ---
    logging.info("Starting prediction on new sentences...")
    nlp = spacy.blank("en")

    def predict_category(sentence, model, text_field, label_field, device):
        model.eval()
        with torch.no_grad():
            tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
            indexed = [text_field.vocab.stoi[t] for t in tokenized]
            length = [len(indexed)]
            tensor = torch.LongTensor(indexed).to(device).unsqueeze(1)
            length_tensor = torch.LongTensor(length)
            
            prediction = model(tensor, length_tensor)
            probabilities = F.softmax(prediction, dim=1)
            
            predicted_class_index = torch.argmax(probabilities).item()
            predicted_class_prob = torch.max(probabilities).item()
            
            return label_field.vocab.itos[predicted_class_index], predicted_class_prob

    test_sentences = [
        "The new iPhone 14 is out and it's amazing!",
        "The stock market crashed today, causing panic among investors.",
        "The latest movie from Christopher Nolan is a masterpiece.",
        "The weather is terrible today, with heavy rain and strong winds."
    ]

    for sentence in test_sentences:
        category, probability = predict_category(sentence, model, TEXT, LABEL, DEVICE)
        logging.info(f"Sentence: '{sentence}' -> Predicted: {category.upper()} (Prob: {probability:.4f})")

    logging.info("Prediction complete.")