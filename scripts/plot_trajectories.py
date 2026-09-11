#!/usr/bin/env python3
"""Plot Figure-3-style trajectory histograms for one or more trained policies.

Each panel shows a 2D hexbin of states across N eval episodes.
Color = visit frequency.

For Pendulum envs (E1/E2/E3) the plot is automatically in (sin θ, cos θ) space
with the excluded zone drawn as a shaded arc.  For any other env it falls back
to plotting two raw observation dimensions (default: obs[0] vs obs[1]).

Usage:
    # Pendulum — auto-detected, zone arc drawn automatically:
    python plot_trajectories.py \\
        runs/_reduced_scope_50k/baseline_E1_p10_seed0_fast \\
        runs/_reduced_scope_50k/lagr_E1_epsilon0.02_seed0_fast

    # merge seeds into one panel:
    python plot_trajectories.py \\
        runs/_reduced_scope_50k/baseline_E1_p10_seed0_fast \\
        runs/_reduced_scope_50k/baseline_E1_p10_seed1_fast \\
        runs/_reduced_scope_50k/lagr_E1_epsilon0.02_seed0_fast \\
        runs/_reduced_scope_50k/lagr_E1_epsilon0.02_seed1_fast \\
        --group "baseline p=10" "baseline p=10" "lagr ε=0.02" "lagr ε=0.02"

    # non-Pendulum env, pick observation dims to plot:
    python plot_trajectories.py runs/my_cartpole_run --x-dim 0 --y-dim 2

--group lets you merge multiple seeds into a single panel (same label = same panel).
Without --group each run gets its own panel.
"""
from __future__ import annotations
import argparse
import glob as _glob
import sys
from collections import defaultdict
from pathlib import Path

# ensure project root is on the path (script lives in scripts/, denrl is in root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Pendulum-specific imports — used only when env is detected as Pendulum
_PENDULUM_ENV_IDS = {"E1", "E2", "E3"}

METHOD_COLOR = {
    "baseline": "#999999",
    "ens":      "#0072B2",
    "bnn":      "#56B4E9",
    "lagr":     "#D55E00",
    "bnn+lagr": "#009E73",
}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# projection: obs -> (x, y) for plotting
# ---------------------------------------------------------------------------

def _is_pendulum(cfg: dict) -> bool:
    return cfg["env"].get("id", "E1") in _PENDULUM_ENV_IDS


def _make_projection(cfg: dict, x_dim: int | None, y_dim: int | None):
    """Return a function obs -> (x, y) and axis labels."""
    if _is_pendulum(cfg) and x_dim is None and y_dim is None:
        # Pendulum obs = [cos θ, sin θ, θ̇]
        def proj(obs):
            return obs[:, 1], obs[:, 0]   # (sin θ, cos θ)
        return proj, "sin θ", "cos θ"

    xi = x_dim if x_dim is not None else 0
    yi = y_dim if y_dim is not None else 1

    def proj(obs):
        return obs[:, xi], obs[:, yi]
    return proj, f"obs[{xi}]", f"obs[{yi}]"


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

def _collect_obs(run_dir: Path, n_episodes: int, seed: int, deterministic: bool = True,
                 swing_up_only: bool = False) -> np.ndarray:
    """Roll out the saved policy; return raw obs array of shape (N, obs_dim)."""
    from denrl.env import make_sim_env, obs_to_theta
    from stable_baselines3 import SAC

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    env = make_sim_env(cfg["env"])
    model = SAC.load(str(run_dir / "policy"), env=None)

    rng = np.random.default_rng(seed)
    max_steps = cfg.get("eval", {}).get("max_steps", 200)
    all_obs = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, _ = env.step(action)
            all_obs.append(obs)
            if swing_up_only and abs(obs_to_theta(obs)) < 0.2:
                break
            if term or trunc:
                break

    env.close()
    return np.array(all_obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# zone overlay (Pendulum-specific)
# ---------------------------------------------------------------------------

def _draw_zone_arc(ax, zone, color="#CC0000", alpha=0.18):
    if hasattr(zone, "zone_type") and getattr(zone, "zone_type") == "box":
        rect = Rectangle(
            (zone.x_min, zone.y_min),
            zone.x_max - zone.x_min,
            zone.y_max - zone.y_min,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(rect)
        return

    def _wedge(lo, hi):
        thetas = np.linspace(lo, hi, 200)
        xs, ys = np.sin(thetas), np.cos(thetas)
        ax.fill(np.concatenate([[0], xs, [0]]),
                np.concatenate([[0], ys, [0]]),
                color=color, alpha=alpha, zorder=2)
        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.6, zorder=3)

    _wedge(zone.lo, zone.hi)
    if getattr(zone, "symmetric", False):
        _wedge(-zone.hi, -zone.lo)


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def _panel(ax, xs: np.ndarray, ys: np.ndarray, label: str, xlabel: str,
           ylabel: str, method: str, pendulum: bool, zone=None):
    color = METHOD_COLOR.get(method, "#333333")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_method", ["#ffffff", color]
    )

    x_pad = (xs.max() - xs.min()) * 0.05 or 0.1
    y_pad = (ys.max() - ys.min()) * 0.05 or 0.1
    extent = [xs.min() - x_pad, xs.max() + x_pad,
              ys.min() - y_pad, ys.max() + y_pad]

    ax.hexbin(xs, ys, gridsize=40, cmap=cmap, linewidths=0.2,
              extent=extent, mincnt=1, bins="log", zorder=1)

    if pendulum and zone is not None:
        _draw_zone_arc(ax, zone)
        # unit circle outline
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.sin(th), np.cos(th), "k-", lw=0.6, alpha=0.3, zorder=4)
        ax.set_xlim(1.15, -1.15)   # reversed x-axis to match paper's sin(θ) orientation
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(label, fontsize=10, color=color, fontweight="bold")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# standalone snapshot — reused by EvalCallback and final eval in run.py
# ---------------------------------------------------------------------------

def plot_trajectory_snapshot(obs: np.ndarray, zone, save_path, title: str = "",
                             method: str = "baseline"):
    """Save a single trajectory hexbin plot from raw obs (N, obs_dim).

    Reuses the same styling as the multi-panel CLI tool.
    """
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    color = METHOD_COLOR.get(method, "#333333")
    cmap = mcolors.LinearSegmentedColormap.from_list("white_to_method", ["#ffffff", color])

    if hasattr(zone, "zone_type") and getattr(zone, "zone_type") == "box":
        xs, ys = obs[:, 0], obs[:, 1]
        ax.hexbin(xs, ys, gridsize=40, cmap=cmap, linewidths=0.2,
                 mincnt=1, bins="log", zorder=1)
        _draw_zone_arc(ax, zone)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_xlim(min(xs.min(), zone.x_min) - 0.2, max(xs.max(), zone.x_max) + 0.2)
        ax.set_ylim(min(ys.min(), zone.y_min) - 0.1, max(ys.max(), zone.y_max) + 0.1)
    else:
        xs, ys = obs[:, 1], obs[:, 0]  # sin θ, cos θ
        ax.hexbin(xs, ys, gridsize=40, cmap=cmap, linewidths=0.2,
                 mincnt=1, bins="log", zorder=1)
        _draw_zone_arc(ax, zone)
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.sin(th), np.cos(th), "k-", lw=0.6, alpha=0.3, zorder=4)
        ax.set_xlim(1.15, -1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel("sin θ", fontsize=9)
        ax.set_ylabel("cos θ", fontsize=9)

    ax.set_title(title, fontsize=9, color=color, fontweight="bold")
    ax.tick_params(labelsize=8)

    from pathlib import Path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    np.save(save_path.with_suffix(".npy"), obs)


# ---------------------------------------------------------------------------
# label helpers
# ---------------------------------------------------------------------------

def _auto_label(cfg: dict) -> str:
    method = cfg.get("method", "?")
    w = cfg.get("weight", {})
    if w.get("type") == "fixed":
        return f"{method}  p={w.get('p', '?')}"
    if w.get("type") == "lagrangian":
        return f"{method}  ε={w.get('epsilon', '?')}"
    return method


def _zone_from_cfg(cfg: dict):
    from denrl.env import as_zone, ZONE_LOW, ZONE_HIGH
    return as_zone(cfg["env"].get("excluded_zone", (ZONE_LOW, ZONE_HIGH)),
                   symmetric=cfg["env"].get("zone_symmetric", False))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Trajectory heatmaps for any trained policy.")
    ap.add_argument("run_dirs", nargs="*", help="paths to run directories")
    ap.add_argument("--group", nargs="*", metavar="LABEL",
                    help="one label per run_dir; same label -> merged panel")
    ap.add_argument("--glob", nargs="*", metavar="PATTERN_OR_LABEL",
                    help="alternating PATTERN LABEL pairs, e.g. "
                         "--glob 'runs/baseline_E1_p10_seed*' 'baseline p=10' "
                         "'runs/lagr_E1_epsilon0.02_seed*' 'lagr ε=0.02'")
    ap.add_argument("--episodes", type=int, default=500,
                    help="eval episodes per run (default 500; use 2000 for paper quality)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--x-dim", type=int, default=None,
                    help="obs index for x-axis (non-Pendulum envs; default 0)")
    ap.add_argument("--y-dim", type=int, default=None,
                    help="obs index for y-axis (non-Pendulum envs; default 1)")
    ap.add_argument("--swing-up-only", action="store_true",
                    help="stop collecting points per episode once upright is reached (|θ| < 0.2); "
                         "reveals left vs right swing-up path without stabilization dominating")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy distribution instead of taking the mean "
                         "(shows spread of behavior, useful for converged policies)")
    ap.add_argument("--out", default="figures/trajectories.png")
    args = ap.parse_args()

    # --- resolve run dirs + labels ---
    run_dirs: list[Path] = []
    labels: list[str] = []

    # positional run_dirs
    for d in args.run_dirs:
        run_dirs.append(Path(d))
        labels.append(None)  # filled below

    # --glob PATTERN LABEL PATTERN LABEL ...
    if args.glob:
        if len(args.glob) % 2 != 0:
            ap.error("--glob requires alternating PATTERN LABEL pairs (even number of arguments)")
        for pattern, label in zip(args.glob[::2], args.glob[1::2]):
            matched = sorted(_glob.glob(pattern))
            if not matched:
                ap.error(f"--glob pattern matched nothing: {pattern}")
            for d in matched:
                run_dirs.append(Path(d))
                labels.append(label)

    if not run_dirs:
        ap.error("provide at least one run directory (positional) or --glob pattern")

    for d in run_dirs:
        if not (d / "policy.zip").exists():
            ap.error(f"no policy.zip in {d}")
        if not (d / "config.yaml").exists():
            ap.error(f"no config.yaml in {d}")

    cfgs = [yaml.safe_load((d / "config.yaml").read_text()) for d in run_dirs]

    # fill labels for positional args (None placeholders)
    if args.group:
        if len(args.group) != len(args.run_dirs):
            ap.error("--group must have the same number of entries as positional run_dirs")
        for i, lbl in enumerate(args.group):
            labels[i] = lbl
    for i, lbl in enumerate(labels):
        if lbl is None:
            labels[i] = _auto_label(cfgs[i])

    groups: dict[str, list] = defaultdict(list)
    for label, d in zip(labels, run_dirs):
        groups[label].append(d)

    # first cfg per group drives the projection and zone
    first_cfg_per_group = {
        label: yaml.safe_load((dirs[0] / "config.yaml").read_text())
        for label, dirs in groups.items()
    }

    print(f"Rolling out {args.episodes} episodes per run...")
    group_obs: dict[str, np.ndarray] = {}
    for label, dirs in groups.items():
        all_obs = []
        for d in dirs:
            obs = _collect_obs(d, args.episodes, args.seed,
                               deterministic=not args.stochastic,
                               swing_up_only=args.swing_up_only)
            all_obs.append(obs)
            print(f"  {d.name}: {len(obs)} steps")
        group_obs[label] = np.concatenate(all_obs)

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, label in zip(axes, groups):
        cfg = first_cfg_per_group[label]
        obs = group_obs[label]
        pendulum = _is_pendulum(cfg)
        proj, xlabel, ylabel = _make_projection(cfg, args.x_dim, args.y_dim)
        xs, ys = proj(obs)
        zone = _zone_from_cfg(cfg) if pendulum else None
        method = cfg.get("method", "baseline")
        _panel(ax, xs, ys, label, xlabel, ylabel, method, pendulum, zone)

    env_id = first_cfg_per_group[next(iter(groups))]["env"].get("id", "")
    subtitle = "red arc = excluded zone" if env_id in _PENDULUM_ENV_IDS else ""
    title = f"Trajectory density  {('(' + subtitle + ')') if subtitle else ''}".strip()
    fig.suptitle(title, fontsize=11, y=1.01)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
