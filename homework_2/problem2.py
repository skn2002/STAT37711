import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 2026
FEATURES = ['StockCode', 'CustomerID']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)


def mean_encoding(fit, apply, target):
    result = np.empty((len(apply), len(FEATURES)), dtype=np.float64)
    fallback = float(np.mean(target))
    for j, name in enumerate(FEATURES):
        means = pd.DataFrame({'key': fit[name].to_numpy(), 'y': target}).groupby('key')['y'].mean()
        result[:, j] = apply[name].map(means).fillna(fallback).to_numpy()
    return result


def encode_features(train, val, test, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    train_values = np.zeros((len(train), 2), dtype=np.float64)
    for fit_idx, hold_idx in folds.split(train):
        train_values[hold_idx] = mean_encoding(train.iloc[fit_idx], train.iloc[hold_idx], y[fit_idx])
    val_values = mean_encoding(train, val, y)
    test_values = mean_encoding(train, test, y)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(np.log1p(train_values))
    val_values = scaler.transform(np.log1p(val_values))
    test_values = scaler.transform(np.log1p(test_values))
    return [a.astype(np.float32) for a in (train_values, val_values, test_values)]


def category_indices(train, val, test):
    matrices = [np.zeros((len(frame), 2), dtype=np.int64) for frame in (train, val, test)]
    sizes = []
    for j, name in enumerate(FEATURES):
        vocab = pd.Index(train[name].unique())
        sizes.append(len(vocab) + 1)
        for matrix, frame in zip(matrices, (train, val, test)):
            matrices_col = pd.Categorical(frame[name], categories=vocab).codes
            matrix[:, j] = matrices_col.astype(np.int64) + 1
    return matrices, sizes


class RetailNet(nn.Module):
    def __init__(self, sizes, use_target):
        super().__init__()
        self.use_target = use_target
        self.embeddings = nn.ModuleList([nn.Embedding(size, 16, padding_idx=0) for size in sizes])
        self.network = nn.Sequential(
            nn.Linear(32 + 2 * int(use_target), 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, category, encoded):
        x = torch.cat([layer(category[:, j]) for j, layer in enumerate(self.embeddings)], dim=1)
        if self.use_target:
            x = torch.cat([x, encoded], dim=1)
        return self.network(x).squeeze(1)


def fit_network(sizes, categories, encoded, y_train, y_val, use_target):
    torch.manual_seed(SEED)
    model = RetailNet(sizes, use_target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(categories[0]), torch.from_numpy(encoded[0]),
        torch.from_numpy(y_train.astype(np.float32))
    )
    loader = DataLoader(dataset, batch_size=16384, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    val_category = torch.from_numpy(categories[1])
    val_encoded = torch.from_numpy(encoded[1])
    val_target = torch.from_numpy(y_val.astype(np.float32))
    best_loss, best_state, best_epoch, stalled = float('inf'), None, 0, 0

    for epoch in range(1, 36):
        model.train()
        for category, te, target in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(category, te), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_category, val_encoded)
            val_loss = nn.functional.mse_loss(val_prediction, val_target).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stalled = 0
        else:
            stalled += 1
        if stalled >= 4:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def scores(true, predicted_log):
    predicted_log = np.maximum(predicted_log, 0.0)
    predicted = np.expm1(predicted_log)
    return (np.sqrt(mean_squared_error(np.log1p(true), predicted_log)),
            mean_absolute_error(true, predicted))


def main():
    data = pd.read_csv(Path(__file__).with_name('retail.csv'), dtype={'StockCode': 'string', 'CustomerID': 'string'})
    train, remaining = train_test_split(data, test_size=0.2, random_state=SEED)
    val, test = train_test_split(remaining, test_size=0.5, random_state=SEED)
    train, val, test = [frame.reset_index(drop=True) for frame in (train, val, test)]
    q_train, q_val, q_test = [frame['Quantity'].to_numpy(dtype=np.float64) for frame in (train, val, test)]
    y_train, y_val = np.log1p(q_train), np.log1p(q_val)

    encoded = encode_features(train, val, test, q_train)
    categories, sizes = category_indices(train, val, test)

    print('Rows:', len(data), '| split:', len(train), len(val), len(test))
    print('Unique categories:', {name: data[name].nunique() for name in FEATURES})
    print('Quantity: median={:.0f}, p99={:.0f}, max={:.0f}'.format(
        data.Quantity.median(), data.Quantity.quantile(0.99), data.Quantity.max()))
    print('Unseen categories (validation/test):',
          [(int((c == 0).sum(axis=0)[0]), int((c == 0).sum(axis=0)[1])) for c in categories[1:]])

    candidates = [100, 200, 400]
    fitted = []
    for steps in candidates:
        model = HistGradientBoostingRegressor(max_iter=steps, max_leaf_nodes=15,
            learning_rate=0.05, l2_regularization=1.0, early_stopping=False,
            random_state=SEED)
        model.fit(encoded[0], y_train)
        loss = mean_squared_error(y_val, model.predict(encoded[1]))
        fitted.append((loss, steps, model))
    val_loss, steps, target_model = min(fitted, key=lambda x: x[0])
    print('Target encoding: iterations={}, val log-MSE={:.6f}'.format(steps, val_loss))

    models = {}
    for name, use_target in [('Embedding', False), ('Combined', True)]:
        model, epoch, loss = fit_network(sizes, categories, encoded, y_train, y_val, use_target)
        models[name] = model
        print('{}: epoch={}, val log-MSE={:.6f}'.format(name, epoch, loss))

    predictions = {'Target encoding': target_model.predict(encoded[2])}
    with torch.no_grad():
        for name, model in models.items():
            predictions[name] = model(torch.from_numpy(categories[2]),
                                      torch.from_numpy(encoded[2])).numpy()
    for name, prediction in predictions.items():
        rmsle, mae = scores(q_test, prediction)
        print('{}: RMSLE={:.6f}, MAE={:.6f}'.format(name, rmsle, mae))


if __name__ == '__main__':
    main()
