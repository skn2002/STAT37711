"""Train and compare RKS and MLP models on the provided toy dataset."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader import HW2Dataset
from models import MLP, RandomKitchenSinks, mlp_sweep, sweep_rks_model
from plotting import plot_surface_2d


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "Fig2"
FIG.mkdir(exist_ok=True)
np.random.seed(2026)
torch.manual_seed(2026)
torch.set_num_threads(1)


def plot_sweep(results, name, parameter):
    for col, split in [(1, "train"), (2, "val")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(results[:, 0], results[:, col], "o-", linewidth=1.5)
        ax.set(xlabel=parameter, ylabel="MSE", title=f"{name}: {split} MSE")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"{name.lower()}_{split}_mse.png", dpi=200)
        plt.close(fig)


def main():
    ds = HW2Dataset.from_csv(ROOT / "train.csv", ROOT / "val.csv", ROOT / "test.csv")
    X_train, y_train = ds.train.X, ds.train.y
    X_val, y_val = ds.val.X, ds.val.y
    X_test = ds.test.X

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                         cmap="viridis", s=18)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="x1", ylabel="x2",
           title="Training data", aspect="equal")
    fig.colorbar(scatter, ax=ax, label="y")
    fig.tight_layout()
    fig.savefig(FIG / "training_data.png", dpi=200)
    plt.close(fig)

    rks_grid = [20, 40, 80, 160, 320]
    _, _, best_D, rks_results = sweep_rks_model(
        rks_grid, X_train, y_train, X_val, y_val)
    plot_sweep(rks_results, "RKS", "Number of random features D")
    rks = RandomKitchenSinks(D=best_D, sigma=1.0,
                             ridge_lambda=1e-3, seed=2026).fit(X_train, y_train)

    width_grid = [8, 16, 32, 64, 128]
    _, _, best_width, mlp_results = mlp_sweep(
        width_grid, X_train, y_train, X_val, y_val)
    plot_sweep(mlp_results, "MLP", "Hidden-layer width")
    torch.manual_seed(2026)
    mlp = MLP(d_in=2, d_hidden=best_width, batch_size=len(X_train)).fit(
        X_train, y_train, X_val, y_val)

    rks_mse = np.mean((rks.predict(X_val) - y_val) ** 2)
    mlp_mse = np.mean((mlp.predict(X_val) - y_val) ** 2)
    np.savetxt(ROOT / "rks_test_predictions.csv", rks.predict(X_test),
               delimiter=",", fmt="%.10f")
    np.savetxt(ROOT / "mlp_test_predictions.csv", mlp.predict(X_test),
               delimiter=",", fmt="%.10f")

    plot_surface_2d(rks.predict, (-3, 3), (-3, 3),
                    title="RKS fitted surface", filename=str(FIG / "rks_surface.png"))
    plt.close()
    plot_surface_2d(mlp, (-3, 3), (-3, 3),
                    title="MLP fitted surface", filename=str(FIG / "mlp_surface.png"))
    plt.close()

    print("RKS sweep (D, train MSE, validation MSE):\n", rks_results)
    print("MLP sweep (width, train MSE, validation MSE):\n", mlp_results)
    print(f"Best RKS: D={best_D}, validation MSE={rks_mse:.6f}")
    print(f"Best MLP: width={best_width}, epoch={mlp.best_epoch_}, "
          f"validation MSE={mlp_mse:.6f}")
    print(f"Saved {len(X_test)} predictions for each model")


if __name__ == "__main__":
    main()
