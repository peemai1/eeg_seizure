import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final classification layer per time step
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, channels, time)
        # -> transpose to (batch_size, time, channels) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM output
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)

        # Predict for each time step
        logits = self.fc(lstm_out)  # (batch_size, seq_len, 1)

        # Remove last dim: (batch_size, seq_len)
        logits = logits.squeeze(-1)

        return logits
