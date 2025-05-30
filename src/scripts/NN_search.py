import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os


class AdvancedMLPModel:
    def __init__(self, data_path, input_dim, output_dim, learning_rate=0.001, device=None):
        self.data_path = data_path
        self.X_Path = self.data_path + "training_X.npy"
        self.Y_Path = self.data_path + "training_Y.npy"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._build_model().to(self.device)

        self.history = None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, self.output_dim),
            nn.Sigmoid(),
            nn.Hardtanh(min_val=0.0, max_val=1.0)
        )
        return model

    def load_and_preprocess_data(self, test_size=0.2):
        X_data = np.load(self.X_Path)
        Y_data = np.load(self.Y_Path)
        self.normalize_data(Y_data)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled,
            self.Y_norm,
            test_size=test_size,
            random_state=42
        )

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.Y_train = torch.tensor(Y_train, dtype=torch.float32).to(self.device)
        self.Y_test = torch.tensor(Y_test, dtype=torch.float32).to(self.device)

    def normalize_data(self, Y_data):
        q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                                      -2.09439510239, -3.05432619099])
        q_max = -q_min

        q_dot_min = np.array([-1.71, -1.74, -1.745, -2.269, -2.443, -3.142, -3.142])
        q_dot_max = -q_dot_min

        self.Y_norm = Y_data.copy()
        self.Y_norm[:, :7] = (Y_data[:, :7] - q_min) / (q_max - q_min)
        self.Y_norm[:, 7:14] = (Y_data[:, 7:14] - q_dot_min) / (q_dot_max - q_dot_min)
        self.Y_norm[:, 14:21] = (Y_data[:, 14:21] - q_min) / (q_max - q_min)
        self.Y_norm[:, 21:] = (Y_data[:, 21:] - q_dot_min) / (q_dot_max - q_dot_min)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test)
            loss = self.loss_function(predictions, self.Y_test)
            print(f"Test Loss: {loss.item()}")
        self.model.train()
        return loss.item()

    def train(self, batch_size=128, epochs=2000, save_interval=100):
        self.model.train()
        self.history = {'train_loss': [], 'val_loss': []}

        dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', ncols=100)
            for batch_X, batch_y in epoch_progress:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(dataloader)
            val_loss = self.evaluate()

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_training_history(f"training_history_epoch_{epoch + 1}.csv")

    def save_training_history(self, filename):
        if self.history:
            log_dir = os.path.join(self.data_path, 'log')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_path = os.path.join(log_dir, filename)
            df = pd.DataFrame(self.history)
            df.to_csv(file_path, index=False)
            print(f"Training history saved to {file_path}")
        else:
            print("No training history available to save.")

    def plot_training_history(self):
        if self.history:
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()
        else:
            print("No training history available.")

    def get_training_history(self):
        if self.history:
            return pd.DataFrame(self.history)
        else:
            print("No training history available.")
            return None

if __name__ == "__main__":
    data_path = "../../../training_data/"
    model = AdvancedMLPModel(data_path=data_path, input_dim=6, output_dim=28)
    model.load_and_preprocess_data(test_size=0.2)
    model.train(batch_size=64, epochs=500)
    model.plot_training_history()

