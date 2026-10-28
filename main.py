"""
backup_full_project.py

COMPLETE working backup for PS-003: Federated Traffic-Flow Prediction with
Communication-Cost Evaluation. Needs data_pipeline.py in the same folder.

Install:   pip install torch numpy matplotlib
Real data: python backup_full_project.py --data PEMS08.npz
Quick test (synthetic data, no download): python backup_full_project.py --synthetic

What it runs (this is the experiment grid the task asks for):
    {centralised, federated IID, federated non-IID} x seeds {0, 1, 2}
Outputs in --out (default: results/):
    results.csv   every logged step of every run
    summary.txt   final MAE/RMSE (mean and std over seeds) and bytes sent
    error_vs_round.png, error_vs_bytes.png, final_error_bars.png

------------------------------------------------------------------------------
PSEUDOCODE (for the mentor demo)
------------------------------------------------------------------------------
split each sensor's timeline: first 70% train, last 30% test (no shuffling)
normalize with mean/std of the training part only
make windows: input = last 12 values, target = next value
clients = one sensor each (non-IID)  OR  pooled windows dealt out randomly (IID)

FedAvg:
    global_model = new small MLP
    repeat for each round:
        for each client k:
            local_model = copy of global_model
            train local_model on client k's data for E epochs
        global_model = sum over k of (n_k / n_total) * local_model_k
        bytes_sent += 2 * clients * parameters * 4        # download + upload
        record MAE and RMSE of global_model on the test windows

Centralised baseline: train one MLP on all training windows pooled together.
------------------------------------------------------------------------------
"""

import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import (load_flow, pick_sensors, build_clients, pooled,
                           mae, rmse, naive_baseline_scores,
                           comm_bytes_per_round)


# ----------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------
def set_all_seeds(seed):
    """Reproducibility: the same seed must give the same run."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(input_len, hidden=32):
    """Compact MLP: input_len past values -> 1 predicted value (~1000 params)."""
    return nn.Sequential(
        nn.Linear(input_len, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, 1),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_local(model, X, y, epochs, lr, batch_size, seed):
    """Standard training loop, updates `model` in place.

    Used by both the centralised baseline and every federated client. A fresh
    Adam optimizer is created on each call in both settings, so the comparison
    treats them identically.
    """
    gen = torch.Generator().manual_seed(seed)
    data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()


@torch.no_grad()
def evaluate(model, X, y, scaler):
    """(MAE, RMSE) in ORIGINAL units (vehicles per 5 min), not normalized."""
    model.eval()
    pred = model(torch.from_numpy(X)).squeeze(1).numpy()
    return mae(scaler.inverse(y), scaler.inverse(pred)), \
        rmse(scaler.inverse(y), scaler.inverse(pred))


def average_weights(client_states, client_sizes):
    """FedAvg aggregation: average weighted by each client's sample count."""
    total = float(sum(client_sizes))
    avg = {}
    for key in client_states[0]:
        avg[key] = sum(state[key] * (n / total)
                       for state, n in zip(client_states, client_sizes))
    return avg


# ----------------------------------------------------------------------------
# The two training procedures
# ----------------------------------------------------------------------------
def run_centralised(clients, scaler, seed, epochs, lr, batch_size, input_len):
    """One model trained on all training windows pooled. Logs after each epoch."""
    set_all_seeds(seed)
    model = build_model(input_len)
    Xtr, ytr = pooled(clients, "train")
    Xte, yte = pooled(clients, "test")
    log = []
    for ep in range(1, epochs + 1):
        train_local(model, Xtr, ytr, 1, lr, batch_size, seed * 1000 + ep)
        m, r = evaluate(model, Xte, yte, scaler)
        log.append(dict(step=ep, mae=m, rmse=r, bytes_cum=0))
    return log


def run_federated(clients, scaler, seed, rounds, local_epochs, lr, batch_size,
                  input_len):
    """FedAvg simulation. Logs MAE, RMSE and cumulative bytes every round."""
    set_all_seeds(seed)
    global_model = build_model(input_len)
    n_params = count_params(global_model)
    sizes = [len(c["y_train"]) for c in clients]
    Xte, yte = pooled(clients, "test")
    log, bytes_cum = [], 0
    for rnd in range(1, rounds + 1):
        states = []
        for i, c in enumerate(clients):
            local = copy.deepcopy(global_model)  # client starts from global
            train_local(local, c["X_train"], c["y_train"], local_epochs, lr,
                        batch_size, seed * 100000 + rnd * 100 + i)
            states.append({k: v.clone() for k, v in local.state_dict().items()})
        global_model.load_state_dict(average_weights(states, sizes))
        bytes_cum += comm_bytes_per_round(n_params, len(clients))
        m, r = evaluate(global_model, Xte, yte, scaler)
        log.append(dict(step=rnd, mae=m, rmse=r, bytes_cum=bytes_cum))
    return log


# ----------------------------------------------------------------------------
# Experiment grid, saving, plotting
# ----------------------------------------------------------------------------
def make_synthetic_flow(T=6000, S=20, seed=0):
    """Fake traffic: daily cycle, different level and amplitude per sensor."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    cols = []
    for k in range(S):
        level = 50 + 25 * k
        amp = 0.2 + 0.02 * k
        cols.append(level * (1 + amp * np.sin(2 * np.pi * t / 288 + k))
                    + rng.normal(0, 0.05 * level, T))
    return np.stack(cols, axis=1).astype(np.float32)


def run_grid(flow, args):
    ids = pick_sensors(flow, args.n_clients, train_frac=args.train_frac)
    print("Chosen sensors:", ids.tolist())
    rows = []
    naive = None
    for seed in args.seeds:
        for partition in ("iid", "noniid"):
            clients, scaler = build_clients(
                flow, ids, input_len=args.input_len, horizon=1,
                train_frac=args.train_frac, partition=partition, seed=seed)
            if naive is None:
                naive = naive_baseline_scores(clients, scaler)

            log = run_federated(clients, scaler, seed, args.rounds,
                                args.local_epochs, args.lr, args.batch_size,
                                args.input_len)
            for entry in log:
                rows.append(dict(method=f"fed_{partition}", seed=seed, **entry))
            print(f"seed {seed} fed_{partition}: final MAE "
                  f"{log[-1]['mae']:.2f}, RMSE {log[-1]['rmse']:.2f}")

            if partition == "noniid":  # pooled data is the same in both settings
                log = run_centralised(clients, scaler, seed,
                                      args.rounds * args.local_epochs, args.lr,
                                      args.batch_size, args.input_len)
                for entry in log:
                    rows.append(dict(method="centralised", seed=seed, **entry))
                print(f"seed {seed} centralised: final MAE "
                      f"{log[-1]['mae']:.2f}, RMSE {log[-1]['rmse']:.2f}")
    return rows, naive


def save_results(rows, naive, args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "step", "mae",
                                          "rmse", "bytes_cum"])
        w.writeheader()
        w.writerows(rows)

    methods = ["centralised", "fed_iid", "fed_noniid"]
    lines = [f"Naive last-value baseline: MAE {naive[0]:.2f}, RMSE {naive[1]:.2f}",
             f"Settings: clients={args.n_clients}, rounds={args.rounds}, "
             f"local_epochs={args.local_epochs}, input_len={args.input_len}, "
             f"seeds={args.seeds}", ""]
    for m in methods:
        last_step = max(r["step"] for r in rows if r["method"] == m)
        fin = [r for r in rows if r["method"] == m and r["step"] == last_step]
        maes = np.array([r["mae"] for r in fin])
        rmses = np.array([r["rmse"] for r in fin])
        line = (f"{m:12s} MAE {maes.mean():.2f} +/- {maes.std():.2f}   "
                f"RMSE {rmses.mean():.2f} +/- {rmses.std():.2f}")
        if m != "centralised":
            line += f"   total bytes {fin[0]['bytes_cum']:,}"
        lines.append(line)
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def make_plots(rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["centralised", "fed_iid", "fed_noniid"]
    labels = {"centralised": "Centralised", "fed_iid": "Federated IID",
              "fed_noniid": "Federated non-IID"}

    def curve(method, key):
        steps = sorted({r["step"] for r in rows if r["method"] == method})
        mat = np.array([[next(r[key] for r in rows if r["method"] == method
                              and r["seed"] == s and r["step"] == st)
                         for st in steps] for s in args.seeds])
        return np.array(steps), mat.mean(axis=0), mat.std(axis=0)

    # 1. Convergence: error against round (epoch for centralised)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ["mae", "rmse"]):
        for m in methods:
            x, mu, sd = curve(m, key)
            ax.plot(x, mu, label=labels[m])
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
        ax.set_xlabel("Round (federated) / epoch (centralised)")
        ax.set_ylabel(key.upper())
        ax.legend()
    fig.suptitle("Convergence (mean and std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_vs_round.png"), dpi=150)
    plt.close(fig)

    # 2. Communication cost: error against cumulative megabytes sent
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("fed_iid", "fed_noniid"):
        _, mb, _ = curve(m, "bytes_cum")
        _, mu, sd = curve(m, "mae")
        ax.plot(mb / 1e6, mu, label=labels[m])
        ax.fill_between(mb / 1e6, mu - sd, mu + sd, alpha=0.2)
    ax.set_xlabel("Cumulative data sent (MB)")
    ax.set_ylabel("MAE")
    ax.set_title("Accuracy vs communication cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_vs_bytes.png"), dpi=150)
    plt.close(fig)

    # 3. Final error comparison with seed spread
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(methods):
        _, mu, sd = curve(m, "mae")
        ax.bar(i, mu[-1], yerr=sd[-1], capsize=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([labels[m] for m in methods], rotation=10)
    ax.set_ylabel("Final MAE")
    ax.set_title("Final test error (mean and std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "final_error_bars.png"), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to PEMS .npz file")
    p.add_argument("--synthetic", action="store_true",
                   help="use fake data to test the program")
    p.add_argument("--n_clients", type=int, default=4)
    p.add_argument("--input_len", type=int, default=12)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.synthetic:
        flow = make_synthetic_flow()
    elif args.data:
        flow = load_flow(args.data)
    else:
        raise SystemExit("Give --data PEMS08.npz or use --synthetic")
    print("Flow array shape (time, sensors):", flow.shape)
    rows, naive = run_grid(flow, args)
    save_results(rows, naive, args)
    make_plots(rows, args)
    print(f"\nSaved results.csv, summary.txt and 3 plots in '{args.out}/'")


if __name__ == "__main__":
    main()