#!/usr/bin/env python3
"""STUB — train the frozen state-transition model(s) (§3, §6).

Fills the gap between the offline dataset and the cost signals / model-based env.
Kept minimal on purpose: swap DeltaDynamics for the paper's LSTM (50 units,
4-step sliding window) when you move past the scaffold.

    python scripts/train_transition.py --env E1 --kind single   -> models/E1_single.pt
    python scripts/train_transition.py --env E2 --kind ensemble --n 5
    python scripts/train_transition.py --env E2 --kind bnn --dropout 0.1

For `ensemble`, N models are trained with different seeds and saved as a dict
{0: state_dict, 1: state_dict, ...} — load_transition() in transition.py reads
that layout.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

import numpy as np
import pandas as pd


def load_xy(env: str):
    df = pd.read_parquet(f"data/{env}_offline.parquet")
    obs = df[[c for c in df.columns if c.startswith("obs")]].to_numpy(np.float32)
    act = df[[c for c in df.columns if c.startswith("act")]].to_numpy(np.float32)
    nxt = df[[c for c in df.columns if c.startswith("next")]].to_numpy(np.float32)
    X = np.concatenate([obs, act], axis=1)
    Y = (nxt - obs).astype(np.float32)  # predict delta
    return X, Y, obs.shape[1], act.shape[1]


def train_one(X, Y, obs_dim, act_dim, seed=0, dropout=0.0, epochs=30):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from denrl.transition import DeltaDynamics

    torch.manual_seed(seed)
    model = DeltaDynamics(obs_dim, act_dim, hidden=64, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y)),
                    batch_size=256, shuffle=True)
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model.state_dict()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="E1")
    ap.add_argument("--kind", choices=["single", "ensemble", "bnn"], default="single")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    X, Y, obs_dim, act_dim = load_xy(args.env)
    Path("models").mkdir(exist_ok=True)

    if args.kind == "ensemble":
        state = {i: train_one(X, Y, obs_dim, act_dim, seed=i, epochs=args.epochs)
                 for i in range(args.n)}
        out = f"models/{args.env}_ensemble.pt"
    elif args.kind == "bnn":
        state = train_one(X, Y, obs_dim, act_dim, seed=0,
                          dropout=max(args.dropout, 0.1), epochs=args.epochs)
        out = f"models/{args.env}_bnn.pt"
    else:
        state = train_one(X, Y, obs_dim, act_dim, seed=0, epochs=args.epochs)
        out = f"models/{args.env}_single.pt"

    torch.save(state, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
