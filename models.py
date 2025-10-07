import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        raise NotImplementedError
    def fit(self, X, y, **kwargs):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

class ESN(BaseModel):
    def __init__(self, input_size, reservoir_size=200, spectral_radius=0.9):
        super().__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.Win = np.random.randn(reservoir_size, input_size)
        W = np.random.randn(reservoir_size, reservoir_size)
        eigs = np.abs(np.linalg.eigvals(W))
        self.W = W / np.max(eigs) * spectral_radius
        self.Wout = None

    def _forward_reservoir(self, X):
        states = np.zeros((X.shape[0], self.reservoir_size))
        for i, x in enumerate(X):
            s = np.zeros(self.reservoir_size)
            for t in range(x.shape[0]):
                s = np.tanh(np.dot(self.Win, x[t]) + np.dot(self.W, s))
            states[i] = s
        return states

    def fit(self, X, y, ridge_param=1e-6):
        states = self._forward_reservoir(X)
        X_aug = np.hstack([states, np.ones((states.shape[0], 1))])
        self.Wout = np.linalg.solve(
            X_aug.T @ X_aug + ridge_param * np.eye(X_aug.shape[1]),
            X_aug.T @ y
        )

    def predict(self, X):
        states = self._forward_reservoir(X)
        X_aug = np.hstack([states, np.ones((states.shape[0], 1))])
        return X_aug @ self.Wout

class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def fit(self, X, y, epochs=20, lr=1e-3, device='cpu'):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X, device='cpu'):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        with torch.no_grad():
            preds = self.forward(X_tensor)
        return preds.cpu().numpy().flatten()

class TransformerModel(BaseModel):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        out = self.transformer(x)
        out = out[-1]
        out = self.fc(out)
        return out

    def fit(self, X, y, epochs=20, lr=1e-3, device='cpu'):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X, device='cpu'):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        with torch.no_grad():
            preds = self.forward(X_tensor)
        return preds.cpu().numpy().flatten()
