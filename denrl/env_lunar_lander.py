"""LunarLander-specific env helpers and density-zone logic."""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class LunarDensityZone:
    """Risky near-ground corridor used as LL's excluded low-density region.

    State layout (LunarLander-v3): [x, y, vx, vy, angle, angular_velocity, leg_l, leg_r].
    """

    def __init__(
        self,
        altitude_max: float = 0.35,
        speed_min: float = 0.90,
        descent_speed_min: float = 0.60,
        tilt_min: float = 0.45,
        angular_speed_min: float = 1.00,
    ):
        self.altitude_max = float(altitude_max)
        self.speed_min = float(speed_min)
        self.descent_speed_min = float(descent_speed_min)
        self.tilt_min = float(tilt_min)
        self.angular_speed_min = float(angular_speed_min)

    def contains(self, obs) -> bool:
        arr = np.asarray(obs, dtype=float).reshape(-1)
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

    def __repr__(self):
        return ("LunarDensityZone(altitude_max={0.altitude_max:.2f}, speed_min={0.speed_min:.2f}, "
                "descent_speed_min={0.descent_speed_min:.2f}, tilt_min={0.tilt_min:.2f}, "
                "angular_speed_min={0.angular_speed_min:.2f})").format(self)


DEFAULT_LUNAR_ZONE = LunarDensityZone()


def as_lunar_zone(zone_cfg=None) -> LunarDensityZone:
    if isinstance(zone_cfg, LunarDensityZone):
        return zone_cfg
    cfg = zone_cfg or {}
    return LunarDensityZone(
        altitude_max=cfg.get("altitude_max", DEFAULT_LUNAR_ZONE.altitude_max),
        speed_min=cfg.get("speed_min", DEFAULT_LUNAR_ZONE.speed_min),
        descent_speed_min=cfg.get("descent_speed_min", DEFAULT_LUNAR_ZONE.descent_speed_min),
        tilt_min=cfg.get("tilt_min", DEFAULT_LUNAR_ZONE.tilt_min),
        angular_speed_min=cfg.get("angular_speed_min", DEFAULT_LUNAR_ZONE.angular_speed_min),
    )


def make_lunar_lander_env(env_cfg: dict) -> gym.Env:
    return gym.make("LunarLander-v3", continuous=bool(env_cfg.get("continuous", False)))
