import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


# RNN Encoder
class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return h_n[-1]


# RNN Decoder
class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, seq_len, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.seq_len = seq_len
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoded):
        repeated_input = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
        output, _ = self.rnn(repeated_input)
        return self.fc(output)


# RNNA Model
class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super(RNNAutoencoder, self).__init__()
        self.encoder = RNNEncoder(input_size, hidden_size)
        self.decoder = RNNDecoder(hidden_size, input_size, seq_len)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


# Train function
def train_autoencoder(model, dataloader, epochs=20, lr=0.001, error_threshold=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x, = batch
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
        if error_threshold and avg_loss < error_threshold:
            print("Stopping early due to low reconstruction error.")
            break


# Sequence creator
def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.float32)


# MAIN
if __name__ == "__main__":
    data_folder = "../Datasets/TSB-AD-U/"
    model_save_folder = "model_weight/"
    os.makedirs(model_save_folder, exist_ok=True)

    seq_len = 10
    batch_size = 32
    input_size = 1
    hidden_size = 16
    epochs = 100

    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            print(f"\n🔧 Training on {filename}")

            try:
                # Load and preprocess
                df = pd.read_csv(file_path)
                values = df['Data'].values.reshape(-1, 1)
                labels = df['Label'].values

                # Normalize
                scaler = MinMaxScaler()
                normalized = scaler.fit_transform(values)

                # Use only normal data
                normal_data = normalized[labels == 0]
                if len(normal_data) <= seq_len:
                    print(f"⚠️ Skipping {filename}: not enough normal data.")
                    continue

                train_sequences = create_sequences(normal_data, seq_len)
                train_loader = DataLoader(TensorDataset(train_sequences), batch_size=batch_size, shuffle=True)

                # Model and training
                model = RNNAutoencoder(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len)
                train_autoencoder(model, train_loader, epochs=epochs, error_threshold=0.001)

                # Save model
                save_path = os.path.join(model_save_folder, f"rnna_model_{filename.replace('.csv', '')}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved model to {save_path}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
