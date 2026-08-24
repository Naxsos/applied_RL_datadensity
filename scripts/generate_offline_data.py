#!/usr/bin/env python3
"""Generate the frozen offline dataset (paper: 2000 episodes, random actions,
episode terminated on entering the excluded zone) -> data/<env>_offline.parquet.

    python scripts/generate_offline_data.py --env E1 --episodes 2000

Columns: obs0..,act0..,next0..  (flat, so run.py's loader picks them up by prefix).
For E2 (uncertainty≠density) replace the env + inject the chaotic band here.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

import numpy as np
import pandas as pd
import gymnasium as gym

from denrl.env import Zone, ChaoticBandPendulum, FixedStart, ZONE_LOW, ZONE_HIGH

# E3 = E1 physics but a wider/shifted danger zone (spec §4: "unknown/shifted
# optimal-p"), so a fixed p tuned on E1's zone is miscalibrated here while
# lagr's epsilon-targeting should self-correct regardless of zone geometry.
E3_ZONE = (np.pi / 4, 3 * np.pi / 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="E1")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-start", action="store_true",
                    help="collect from the resting position theta=pi instead of gym's uniform "
                         "reset. NOT the default -- see the note at the reset call below.")
    ap.add_argument("--symmetric-zone", action="store_true",
                    help="mirror the zone onto both sides of the swing (legacy). Forced on "
                         "for E2, whose frozen dataset and transition models were built that "
                         "way; regenerating E2 one-sided would invalidate models/E2_*.pt.")
    args = ap.parse_args()

    env = gym.make("Pendulum-v1")
    lo, hi = ZONE_LOW, ZONE_HIGH
    symmetric = args.symmetric_zone
    if args.env == "E2":
        # same chaotic-band dynamics the sim env uses -> transition models learn
        # high variance in the dense band (Region B)
        env = ChaoticBandPendulum(env, seed=args.seed)
        symmetric = True
    elif args.env == "E3":
        lo, hi = E3_ZONE
    zone = Zone(lo, hi, symmetric=symmetric)
    # Data collection deliberately starts uniformly over the circle, even though the
    # AGENT starts at rest (env.start_state, spec §3). The offline set has to cover the
    # state space for the excluded zone to show up as *the* hole in it: random actions
    # from theta=pi never reach the upper half at all (measured: 0.0% of transitions
    # above |theta|=60 deg, both sides), which would make the KDE cost penalize every
    # route to upright equally and leave the agent no reason to prefer the safe one.
    if args.fixed_start:
        env = FixedStart(env, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    env.action_space.seed(args.seed)   # reset(seed=) does not cover action_space.sample();
                                       # without this the dataset size drifts between runs
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        for t in range(args.max_steps):
            action = env.action_space.sample()
            nxt, _, term, trunc, _ = env.step(action)
            if zone.contains(nxt):  # paper: terminate on entering the low-density zone
                break
            rows.append(np.concatenate([obs, action, nxt]))
            obs = nxt
            if term or trunc:
                break

    arr = np.array(rows)
    n_obs, n_act = 3, 1
    cols = ([f"obs{i}" for i in range(n_obs)] +
            [f"act{i}" for i in range(n_act)] +
            [f"next{i}" for i in range(n_obs)])
    df = pd.DataFrame(arr, columns=cols)
    Path("data").mkdir(exist_ok=True)
    path = f"data/{args.env}_offline.parquet"
    df.to_parquet(path)
    print(f"wrote {path}: {len(df)} transitions from {args.episodes} episodes, {zone}")


if __name__ == "__main__":
    main()
