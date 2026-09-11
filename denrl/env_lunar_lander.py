"""LunarLander-specific env helpers and density-zone logic."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


def _default_lunar_zone_cfg():
    """Return the canonical LL box-zone config from the repo config directory."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "ll_box_zone.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing LL box-zone config: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid LL box-zone config: {cfg_path}")
    return data


class LunarZone:
    """LL excluded low-density region in state space.

    Supports two modes:
    - box: compact rectangular region in position space (x, y), avoidable by steering left/right
    - corridor: legacy near-ground risky corridor based on speed/altitude/attitude thresholds
    """

    def __init__(
        self,
        zone_type: str = "box",
        x_range=None,
        y_range=None,
        altitude_max: float = 0.21,
        speed_min: float = 0.72,
        descent_speed_min: float = 0.32,
        tilt_min: float = 0.47,
        angular_speed_min: float = 0.25,
    ):
        default_box = _default_lunar_zone_cfg()
        x_range = default_box.get("x", (-0.5, 0.5)) if x_range is None else x_range
        y_range = default_box.get("y", (0.65, 0.85)) if y_range is None else y_range
        self.zone_type = str(zone_type).lower()
        if self.zone_type in ("box", "rectangle", "rect", "position_box"):
            self.zone_type = "box"
            self.x_min, self.x_max = float(x_range[0]), float(x_range[1])
            self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        elif self.zone_type in ("corridor", "density", "legacy", "threshold"):
            self.zone_type = "corridor"
            self.altitude_max = float(altitude_max)
            self.speed_min = float(speed_min)
            self.descent_speed_min = float(descent_speed_min)
            self.tilt_min = float(tilt_min)
            self.angular_speed_min = float(angular_speed_min)
        else:
            raise ValueError(f"unknown zone_type {self.zone_type}")

    def contains(self, obs) -> bool:
        arr = np.asarray(obs, dtype=float).reshape(-1)
        if self.zone_type == "box":
            if arr.shape[0] < 2:
                return False
            x, y = float(arr[0]), float(arr[1])
            return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)
        else:  # corridor
            if arr.shape[0] < 6:
                return False
            y = float(arr[1])
            vx, vy = float(arr[2]), float(arr[3])
            angle = abs(float(arr[4]))
            ang_vel = abs(float(arr[5]))
            speed = float(np.hypot(vx, vy))
            descent_speed = max(0.0, -vy)
            near_ground = y <= self.altitude_max
            unstable_descent = speed >= self.speed_min and descent_speed >= self.descent_speed_min
            unstable_attitude = angle >= self.tilt_min or ang_vel >= self.angular_speed_min
            return near_ground and (unstable_descent or unstable_attitude)

    def __eq__(self, other):
        o = as_lunar_zone(other) if not isinstance(other, LunarZone) else other
        if self.zone_type != o.zone_type:
            return False
        if self.zone_type == "box":
            return (self.x_min, self.x_max, self.y_min, self.y_max) == (o.x_min, o.x_max, o.y_min, o.y_max)
        else:
            return (
                self.altitude_max, self.speed_min, self.descent_speed_min, self.tilt_min, self.angular_speed_min
            ) == (o.altitude_max, o.speed_min, o.descent_speed_min, o.tilt_min, o.angular_speed_min)

    def __repr__(self):
        if self.zone_type == "box":
            return f"LunarZone(type=box, x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}])"
        else:
            return (
                f"LunarZone(type=corridor, altitude_max={self.altitude_max:.2f}, speed_min={self.speed_min:.2f}, "
                f"descent_speed_min={self.descent_speed_min:.2f}, tilt_min={self.tilt_min:.2f}, "
                f"angular_speed_min={self.angular_speed_min:.2f})"
            )


default_box_cfg = _default_lunar_zone_cfg()
DEFAULT_LUNAR_ZONE = LunarZone(
    zone_type="box",
    x_range=tuple(default_box_cfg.get("x", (-0.5, 0.5))),
    y_range=tuple(default_box_cfg.get("y", (0.65, 0.85))),
)
LEGACY_LUNAR_ZONE = LunarZone(zone_type="corridor")  # for backwards compatibility


def as_lunar_zone(zone_cfg=None) -> LunarZone:
    """Parse zone config into a LunarZone, defaulting to the canonical box config."""
    if isinstance(zone_cfg, LunarZone):
        return zone_cfg
    cfg = zone_cfg or _default_lunar_zone_cfg()
    if isinstance(cfg, dict):
        kind = str(cfg.get("kind") or cfg.get("type") or "").lower()
        if kind in ("box", "rectangle", "rect", "position_box"):
            x = cfg.get("x", cfg.get("x_range", (-0.5, 0.5)))
            y = cfg.get("y", cfg.get("y_range", (0.65, 0.85)))
            return LunarZone(zone_type="box", x_range=tuple(x), y_range=tuple(y))
        if any(k in cfg for k in (
            "x_min", "x_max", "y_min", "y_max",
            "altitude_max", "speed_min", "descent_speed_min", "tilt_min", "angular_speed_min",
        )):
            if any(k in cfg for k in ("altitude_max", "speed_min", "descent_speed_min", "tilt_min", "angular_speed_min")):
                return LunarZone(
                    zone_type="corridor",
                    altitude_max=cfg.get("altitude_max", 0.21),
                    speed_min=cfg.get("speed_min", 0.72),
                    descent_speed_min=cfg.get("descent_speed_min", 0.32),
                    tilt_min=cfg.get("tilt_min", 0.47),
                    angular_speed_min=cfg.get("angular_speed_min", 0.25),
                )
            else:
                return LunarZone(
                    zone_type="box",
                    x_range=(cfg.get("x_min", -0.5), cfg.get("x_max", 0.5)),
                    y_range=(cfg.get("y_min", 0.65), cfg.get("y_max", 0.85)),
                )
    return DEFAULT_LUNAR_ZONE


def make_lunar_lander_env(env_cfg: dict) -> gym.Env:
    return gym.make("LunarLander-v3", continuous=bool(env_cfg.get("continuous", False)))
