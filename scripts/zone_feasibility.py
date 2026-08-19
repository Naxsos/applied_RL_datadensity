#!/usr/bin/env python3
"""Is the excluded zone avoidable at all? -- exhaustive feasibility check.

    python scripts/zone_feasibility.py                 # E1 and E3 paper zones
    python scripts/zone_feasibility.py --zone 45 135   # any band, in degrees

Answers the question the metrics cannot: when every trained policy shows
zone_visit_rate = 1.0, is that the agent failing to find the detour, or is the
detour physically absent? RL runs cannot separate those two.

Method: forward BFS over a discretized (theta, theta_dot) grid using the exact
Pendulum-v1 dynamics, with zone states deleted from the graph. Grid rounding makes
the search *optimistic*, so an unreachable verdict is strong evidence of genuine
infeasibility, while a reachable verdict is only a candidate -- which is why the
path is then re-simulated on the true continuous dynamics, and a PD controller is
appended to check the pendulum can also be *held* up without falling back through
the zone.

Result on the paper's setup (see EXPERIMENT_SPEC.md §4):
  E1 [72, 108] deg -> FEASIBLE: swing-up + balance with 0 zone steps, 155/200
                     steps upright. So E1's constraint is satisfiable and the
                     trained policies' 1.0 visit rate is a method gap.
  E3 [45, 135] deg -> INFEASIBLE: upright is unreachable without entering the
                     zone. Any policy that solves E3 must violate the constraint,
                     so E3 cannot test the Lagrangian's auto-tuning claim as
                     currently defined.
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denrl.env import ZONE_LOW, ZONE_HIGH   # noqa: E402

# Pendulum-v1 constants (gymnasium/envs/classic_control/pendulum.py)
G, M, L, DT, MAX_TORQUE, MAX_SPEED = 10.0, 1.0, 1.0, 0.05, 2.0, 8.0
N_TH, N_TD = 721, 641                      # 0.5 deg x 0.025 rad/s
TORQUES = np.arange(-MAX_TORQUE, MAX_TORQUE + 1e-9, 0.25)
UPRIGHT = 0.2                              # |theta| counted as upright (metrics.py)

th_grid = np.linspace(-np.pi, np.pi, N_TH, endpoint=False)
td_grid = np.linspace(-MAX_SPEED, MAX_SPEED, N_TD)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def step(th, td, u):
    td = np.clip(td + (3 * G / (2 * L) * np.sin(th) + 3.0 / (M * L ** 2) * u) * DT,
                 -MAX_SPEED, MAX_SPEED)
    return wrap(th + td * DT), td


def build_successors():
    """succ[k, s] = grid index reached from state s under torque TORQUES[k]."""
    TH, TD = np.meshgrid(th_grid, td_grid, indexing="ij")
    th, td = TH.ravel(), TD.ravel()
    succ = np.empty((len(TORQUES), th.size), dtype=np.int32)
    for k, u in enumerate(TORQUES):
        nth, ntd = step(th, td, u)
        i = np.rint((nth + np.pi) / (2 * np.pi) * N_TH).astype(np.int64) % N_TH
        j = np.rint((ntd + MAX_SPEED) / (2 * MAX_SPEED) * (N_TD - 1)).astype(np.int64)
        succ[k] = i * N_TD + j
    return succ


def bfs(zone, succ, max_depth=400):
    """Shortest zone-free path from rest to upright. Returns (actions | None, n_reachable)."""
    TH, _ = np.meshgrid(th_grid, td_grid, indexing="ij")
    th_flat = TH.ravel()
    blocked = (th_flat >= zone[0]) & (th_flat <= zone[1])
    goal = np.abs(th_flat) < UPRIGHT
    start = int(np.argmin(np.abs(th_grid - (np.pi - 2 * np.pi / N_TH))) * N_TD
                + np.argmin(np.abs(td_grid)))

    seen = np.zeros(th_flat.size, bool); seen[start] = True
    parent = np.full(th_flat.size, -1, np.int64)
    pact = np.full(th_flat.size, -1, np.int8)
    frontier = np.array([start], np.int64)

    for _ in range(max_depth):
        reached = []
        for k in range(len(TORQUES)):
            nxt = succ[k, frontier]
            fresh = ~seen[nxt] & ~blocked[nxt]
            if not fresh.any():
                continue
            new, src = nxt[fresh], frontier[fresh]
            uniq, first = np.unique(new, return_index=True)   # BFS: first arrival wins
            parent[uniq], pact[uniq], seen[uniq] = src[first], k, True
            reached.append(uniq)
        if not reached:
            return None, int(seen.sum())
        frontier = np.unique(np.concatenate(reached))
        hit = frontier[goal[frontier]]
        if hit.size:
            acts, cur = [], int(hit[0])
            while parent[cur] >= 0:
                acts.append(float(TORQUES[pact[cur]]))
                cur = int(parent[cur])
            return np.array(acts[::-1]), int(seen.sum())
    return None, int(seen.sum())


def simulate(actions, zone, horizon=200, pd_gains=None):
    """Replay on the exact continuous dynamics; optionally PD-balance after the swing-up."""
    th, td, zone_steps, upright_steps = np.pi, 0.0, 0, 0
    thetas = []
    for t in range(horizon):
        if t < len(actions):
            u = actions[t]
        elif pd_gains is not None:
            kp, kd = pd_gains
            u = float(np.clip(-kp * th - kd * td, -MAX_TORQUE, MAX_TORQUE))
        else:
            u = 0.0
        th, td = step(th, td, u)
        thetas.append(th)
        zone_steps += int(zone[0] <= th <= zone[1])
        upright_steps += int(abs(th) < UPRIGHT)
    return zone_steps, upright_steps, np.array(thetas)


def tune_pd(actions, zone):
    """Smallest-zone-contact PD gains for holding the pendulum up after the swing-up."""
    best = None
    for kp, kd in itertools.product([2, 4, 6, 8, 10, 14, 20], [0.5, 1, 2, 3, 4, 6]):
        zs, us, _ = simulate(actions, zone, pd_gains=(kp, kd))
        if best is None or (zs, -us) < (best[0], -best[1]):
            best = (zs, us, kp, kd)
    return best


def report(name, zone, succ, save_to: Path | None = None):
    lo, hi = np.degrees(zone)
    acts, n_reach = bfs(zone, succ)
    if acts is None:
        print(f"{name} [{lo:.0f}, {hi:.0f}] deg -> INFEASIBLE: upright unreachable without "
              f"entering the zone ({n_reach} of {N_TH*N_TD} states reachable).")
        print("   Any policy that solves the task must violate the constraint, so "
              "zone_visit_rate can only measure whether it gave up.")
        return False
    # swing-up phase only: the torque sequence ends at the top, and with no
    # controller after it the pendulum simply falls back down through the zone --
    # counting those steps would blame the plan for the missing balancer
    swing_zs, _, _ = simulate(acts, zone, horizon=len(acts))
    pd_zs, pd_us, kp, kd = tune_pd(acts, zone)
    print(f"{name} [{lo:.0f}, {hi:.0f}] deg -> FEASIBLE: swing-up in {len(acts)} steps "
          f"({len(acts)*DT:.2f}s), {swing_zs} zone steps on the exact dynamics.")
    print(f"   full 200-step episode with PD balancing (kp={kp}, kd={kd}): "
          f"{pd_zs} zone steps, {pd_us}/200 steps upright.")
    if save_to is not None:
        np.save(save_to, acts)
        print(f"   reference torque sequence -> {save_to}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", nargs=2, type=float, metavar=("LO_DEG", "HI_DEG"),
                    help="check a custom band instead of the E1/E3 defaults")
    args = ap.parse_args()

    print("building successor table...", flush=True)
    succ = build_successors()
    if args.zone:
        report("custom", np.radians(args.zone), succ)
        return
    report("E1", (ZONE_LOW, ZONE_HIGH), succ, save_to=Path("data/reference_safe_swingup_E1.npy"))
    report("E3", (np.pi / 4, 3 * np.pi / 4), succ)


if __name__ == "__main__":
    main()
