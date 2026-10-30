from pathlib import Path
import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


SEED = 2026
FEATURES = [
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'min_of_day', 'day_of_week', 'day_of_year',
]
TARGET = 'trip_duration'


class GaussianMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.network(x)
        mean = output[:, 0]
        std = F.softplus(output[:, 1]) + 1e-4
        return mean, std


def gaussian_nll(y, mean, std):
    return (0.5 * math.log(2 * math.pi)
            + torch.log(std) + 0.5 * ((y - mean) / std).square()).mean()


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        mean, std = model(x)
        return gaussian_nll(y, mean, std).item()


def predict(model, x, target_mean, target_std):
    model.eval()
    with torch.no_grad():
        mean, std = model(x)
    return (mean.numpy() * target_std + target_mean,
            std.numpy() * target_std)


def metrics(y, mean, std):
    probability = ndtr((45.0 - mean) / std)
    label = (y < 45.0).astype(float)
    return {
        'NLL': float(np.mean(0.5 * np.log(2 * np.pi)
                             + np.log(std) + 0.5 * ((y - mean) / std) ** 2)),
        'MAE': float(np.mean(np.abs(y - mean))),
        'RMSE': float(np.sqrt(np.mean((y - mean) ** 2))),
        'Brier': float(np.mean((probability - label) ** 2)),
        'Accuracy': float(np.mean((probability >= 0.5) == label)),
    }


def main(data_path):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    if str(data_path).endswith('.parquet'):
        data = pd.read_parquet(data_path, columns=FEATURES + [TARGET])
    else:
        data = pd.read_csv(data_path, usecols=FEATURES + [TARGET])

    x = data[FEATURES].to_numpy(dtype=np.float32)
    y = data[TARGET].to_numpy(dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()

    indices = np.arange(len(y))
    train_val, test = train_test_split(indices, test_size=0.15, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85,
                                  random_state=SEED)
    x_train, x_val, x_test = x[train], x[val], x[test]
    y_train, y_val, y_test = y[train], y[val], y[test]

    target_mean = y_train.mean()
    target_std = y_train.std()
    y_train_scaled = (y_train - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std

    model = GaussianMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train)
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)

    train_loss, val_loss = [], []
    best_loss, best_epoch, best_state = float('inf'), 0, None
    patience = 10
    for epoch in range(1, 41):
        model.train()
        order = torch.randperm(len(train))
        epoch_loss = 0.0
        for batch in order.split(4096):
            optimizer.zero_grad()
            mean, std = model(x_train_t[batch])
            loss = gaussian_nll(y_train_t[batch], mean, std)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        tr = epoch_loss / len(train) + math.log(target_std)
        va = evaluate(model, x_val_t, y_val_t) + math.log(target_std)
        train_loss.append(tr)
        val_loss.append(va)
        if va < best_loss - 1e-4:
            best_loss, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
        print(f'Epoch {epoch:2d}: train NLL {tr:.4f}, val NLL {va:.4f}', flush=True)

    model.load_state_dict(best_state)
    mlp_mean, mlp_std = predict(model, x_test_t, target_mean, target_std)
    constant_mean = np.full_like(y_test, target_mean)
    constant_std = np.full_like(y_test, target_std)

    linear = Ridge(alpha=1.0).fit(x_train, y_train)
    linear_std = np.sqrt(np.mean((y_train - linear.predict(x_train)) ** 2))
    linear_mean = linear.predict(x_test)

    results = {
        'Constant Gaussian': metrics(y_test, constant_mean, constant_std),
        'Linear Gaussian': metrics(y_test, linear_mean,
                                   np.full_like(y_test, linear_std)),
        'Gaussian MLP': metrics(y_test, mlp_mean, mlp_std),
    }

    output = Path('Fig2')
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label='Training')
    ax.plot(epochs, val_loss, label='Validation')
    ax.set(xlabel='Epoch', ylabel='Gaussian NLL')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'problem1_training.png', dpi=200)
    plt.close(fig)

    summary = {
        'seed': SEED,
        'samples': {'train': len(train), 'validation': len(val), 'test': len(test)},
        'test_fraction_under_45': float(np.mean(y_test < 45)),
        'best_epoch': best_epoch,
        'best_validation_nll': best_loss,
        'test_metrics': results,
    }
    (output / 'problem1_results.json').write_text(json.dumps(summary, indent=2))

    print(f'\nSplit: train={len(train)}, val={len(val)}, test={len(test)}')
    print(f'Best epoch: {best_epoch}, validation NLL: {best_loss:.5f}')
    print(f'Test fraction below 45 min: {summary["test_fraction_under_45"]:.6f}')
    print(f'{"Model":20s} {"NLL":>9s} {"MAE":>9s} {"RMSE":>9s} '
          f'{"Brier":>10s} {"Accuracy":>10s}')
    for name, scores in results.items():
        print(f'{name:20s} {scores["NLL"]:9.4f} {scores["MAE"]:9.4f} '
              f'{scores["RMSE"]:9.4f} {scores["Brier"]:10.6f} '
              f'{scores["Accuracy"]:10.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='taxi_trips.parquet')
    args = parser.parse_args()
    main(args.data)
