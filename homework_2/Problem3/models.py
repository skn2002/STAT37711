"""Random Kitchen Sinks and one-hidden-layer MLP for regression."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class RandomKitchenSinks:
    def __init__(self, D=1000, sigma=1.0, ridge_lambda=1e-3, seed=2026):
        if D <= 0 or D % 2:
            raise ValueError("D must be a positive even integer")
        self.D = D
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.seed = seed

    def _features(self, X):
        angles = np.asarray(X) @ self.W_ + self.b_
        features = np.stack((np.sin(angles), np.cos(angles)), axis=-1)
        return np.column_stack((np.ones(len(X)), features.reshape(len(X), self.D)))

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y).reshape(-1)
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.normal(0, 1 / self.sigma, size=(X.shape[1], self.D // 2))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.D // 2)
        phi = self._features(X)
        penalty = np.eye(self.D + 1)
        penalty[0, 0] = 0  # Do not penalize the intercept.
        self.coef_ = np.linalg.solve(
            phi.T @ phi + len(X) * self.ridge_lambda * penalty, phi.T @ y
        )
        return self

    def predict(self, X):
        return self._features(X) @ self.coef_


def sweep_rks_model(sweep_D, X_train, y_train, X_val, y_val):
    results = []
    best_model, best_val_mse, best_D = None, float("inf"), None
    for D in sweep_D:
        model = RandomKitchenSinks(D=D, seed=2026).fit(X_train, y_train)
        train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
        results.append((D, train_mse, val_mse))
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_D = model, val_mse, D
    return best_model, best_val_mse, best_D, np.asarray(results)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_out=1, lr=1e-3,
                 batch_size=128, activation="relu", device="cpu"):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), activations[activation](),
            nn.Linear(d_hidden, d_out)
        )
        self.to(self.device)

    def forward(self, X):
        return self.net(X)

    def fit(self, X, y, X_val, y_val, epochs=1800, patience=180):
        train_X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        val_X = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        val_y = torch.as_tensor(y_val, dtype=torch.float32, device=self.device).reshape(-1, 1)
        dataset = TensorDataset(train_X, train_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(2026))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_loss, stale, best_state = float("inf"), 0, None
        for epoch in range(epochs):
            self.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self(batch_X), batch_y)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(self(val_X), val_y).item()
            if val_loss < best_loss - 1e-7:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
        self.load_state_dict(best_state)
        self.eval()
        self.best_epoch_ = epoch + 1 - stale
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self(X).cpu().numpy().reshape(-1)


def mlp_sweep(sweep_values, X_train, y_train, X_val, y_val):
    results = []
    best_model, best_val_mse, best_width = None, float("inf"), None
    for width in sweep_values:
        torch.manual_seed(2026)
        model = MLP(d_in=X_train.shape[1], d_hidden=int(width), batch_size=len(X_train))
        model.fit(X_train, y_train, X_val, y_val)
        train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
        results.append((width, train_mse, val_mse))
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_width = model, val_mse, int(width)
    return best_model, best_val_mse, best_width, np.asarray(results)
