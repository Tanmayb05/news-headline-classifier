# src/model.py
import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    The Bidirectional LSTM model for text classification.
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=2,
                           bidirectional=True,
                           dropout=dropout_rate if 2 > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        # Pack sequence to handle variable length inputs efficiently
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)
