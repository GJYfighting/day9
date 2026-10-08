#!/usr/bin/env python3
"""Day6 traditional controller plus a bounded, filtered SAC residual."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import numpy as np
from stable_baselines3 import SAC
import yaml

from day4_env import Day4GraspEnv


OBSERVATION_SHAPE = (10,)
ACTION_SHAPE = (4,)


class ActionGuardError(RuntimeError):
    """Raised before the wrapped environment can receive an unsafe action."""


@dataclass(frozen=True)
class ActionDecision:
    base_action: np.ndarray
    raw_residual: np.ndarray
    mapped_residual: np.ndarray
    filtered_residual: np.ndarray
    final_action: np.ndarray
    residual_enabled: bool
    residual_valid: bool
    fallback_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_action": self.base_action.tolist(),
            "raw_residual": self.raw_residual.tolist(),
            "mapped_residual": self.mapped_residual.tolist(),
            "filtered_residual": self.filtered_residual.tolist(),
            "final_action": self.final_action.tolist(),
            "residual_enabled": self.residual_enabled,
            "residual_valid": self.residual_valid,
            "fallback_reason": self.fallback_reason,
        }


class ResidualActionController:
    """Pure action fusion; this class never publishes robot commands."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        model: Any | None = None,
        load_model: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.root = Path(__file__).resolve().parent
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.d4 = self.config["day4"]
        self.d6 = self.config["day6"]

        final_m = float(self.d4["action_limit_m"])
        final_yaw = math.radians(float(self.d4["yaw_action_limit_deg"]))
        self.final_limit = np.asarray(
            [final_m, final_m, final_m, final_yaw], dtype=np.float32
        )
        self.final_low = -self.final_limit
        self.final_high = self.final_limit

        residual_m = float(self.d6["residual_limit_m"])
        residual_yaw = math.radians(float(self.d6["residual_yaw_limit_deg"]))
        self.residual_limit = np.asarray(
            [residual_m, residual_m, residual_m, residual_yaw], dtype=np.float32
        )
        self.alpha = float(self.d6["low_pass_alpha"])
        self.base_gain = float(self.d4["smoke"]["policy_gain"])
        self.deterministic = bool(self.d6.get("deterministic", True))
        self.enabled = bool(self.d6["enable_residual"])

        if not bool(np.isfinite(self.final_limit).all()) or bool(
            np.any(self.final_limit <= 0.0)
        ):
            raise ValueError("final action limits must be finite and positive")
        if not bool(np.isfinite(self.residual_limit).all()) or bool(
            np.any(self.residual_limit <= 0.0)
        ):
            raise ValueError("residual limits must be finite and positive")
        if not bool(np.all(self.residual_limit < self.final_limit)):
            raise ValueError("every residual limit must be smaller than its final limit")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("low_pass_alpha must be in (0, 1]")

        configured_model = Path(str(self.d6["model_path"]))
        if not configured_model.is_absolute():
            configured_model = self.root / configured_model
        self.model_path = configured_model.resolve()
        try:
            self.model_path.relative_to(self.root)
        except ValueError as exception:
            raise ValueError("model_path must resolve inside the Day6 root") from exception

        self.model: Any | None = model
        self.model_load_error = ""
        self.model_low = self.final_low.copy()
        self.model_high = self.final_high.copy()
        self.model_observation_low: np.ndarray | None = None
        self.model_observation_high: np.ndarray | None = None

        if self.model is None and load_model:
            try:
                self.model = SAC.load(self.model_path, device="cpu")
            except Exception as exception:  # Safe base-only fallback.
                self.model_load_error = f"{type(exception).__name__}: {exception}"
                self.model = None

        if self.model is not None:
            try:
                self._validate_model_spaces()
            except Exception as exception:  # Safe base-only fallback.
                self.model_load_error = f"{type(exception).__name__}: {exception}"
                self.model = None

        self._filtered = np.zeros(ACTION_SHAPE, dtype=np.float32)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and not self.model_load_error

    def _validate_model_spaces(self) -> None:
        observation_space = self.model.observation_space
        action_space = self.model.action_space
        if tuple(observation_space.shape) != OBSERVATION_SHAPE:
            raise ValueError(f"model observation shape is {observation_space.shape}")
        if tuple(action_space.shape) != ACTION_SHAPE:
            raise ValueError(f"model action shape is {action_space.shape}")

        observation_low = np.asarray(observation_space.low, dtype=np.float32)
        observation_high = np.asarray(observation_space.high, dtype=np.float32)
        model_low = np.asarray(action_space.low, dtype=np.float32)
        model_high = np.asarray(action_space.high, dtype=np.float32)
        arrays = (observation_low, observation_high, model_low, model_high)
        if not all(bool(np.isfinite(array).all()) for array in arrays):
            raise ValueError("model spaces contain a non-finite bound")
        if bool(np.any(model_high <= model_low)):
            raise ValueError("model action bounds are invalid")
        if not (
            np.allclose(model_low, self.final_low, rtol=0.0, atol=1e-7)
            and np.allclose(model_high, self.final_high, rtol=0.0, atol=1e-7)
        ):
            raise ValueError("model action bounds do not match the Day5 action space")

        self.model_low = model_low
        self.model_high = model_high
        self.model_observation_low = observation_low
        self.model_observation_high = observation_high

    @staticmethod
    def validate_observation(observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != OBSERVATION_SHAPE:
            raise ActionGuardError(f"observation must have shape {OBSERVATION_SHAPE}")
        if not bool(np.isfinite(array).all()):
            raise ActionGuardError("observation contains NaN or Inf")
        return array

    def compute_base_action(self, observation: np.ndarray) -> np.ndarray:
        array = self.validate_observation(observation)
        yaw_error = math.atan2(float(array[8]), float(array[9]))
        raw = np.asarray(
            [
                self.base_gain * float(array[0]),
                self.base_gain * float(array[1]),
                self.base_gain * float(array[2]),
                self.base_gain * yaw_error,
            ],
            dtype=np.float32,
        )
        return np.clip(raw, self.final_low, self.final_high).astype(
            np.float32, copy=False
        )

    def reset_episode(self) -> None:
        self._filtered = np.zeros(ACTION_SHAPE, dtype=np.float32)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled != self.enabled or not enabled:
            self.reset_episode()
        self.enabled = enabled

    def _infer_raw_residual(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, bool, str]:
        if not self.enabled:
            self.reset_episode()
            return np.zeros(ACTION_SHAPE, dtype=np.float32), True, "disabled"
        if self.model is None:
            self.reset_episode()
            reason = self.model_load_error or "model unavailable"
            return np.zeros(ACTION_SHAPE, dtype=np.float32), False, reason

        try:
            prediction = self.model.predict(
                observation,
                deterministic=self.deterministic,
            )
            raw_value = prediction[0] if isinstance(prediction, tuple) else prediction
            raw = np.asarray(raw_value, dtype=np.float32)
        except Exception as exception:
            self.reset_episode()
            return (
                np.zeros(ACTION_SHAPE, dtype=np.float32),
                False,
                f"inference {type(exception).__name__}: {exception}",
            )

        if raw.shape != ACTION_SHAPE:
            self.reset_episode()
            return (
                np.zeros(ACTION_SHAPE, dtype=np.float32),
                False,
                f"model action shape is {raw.shape}",
            )
        if not bool(np.isfinite(raw).all()):
            self.reset_episode()
            return (
                np.zeros(ACTION_SHAPE, dtype=np.float32),
                False,
                "model action contains NaN or Inf",
            )
        return raw, True, ""

    def _map_residual(self, raw: np.ndarray) -> np.ndarray:
        clipped_model_action = np.clip(raw, self.model_low, self.model_high)
        normalized = 2.0 * (
            (clipped_model_action - self.model_low)
            / (self.model_high - self.model_low)
        ) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        mapped = normalized * self.residual_limit
        return np.clip(mapped, -self.residual_limit, self.residual_limit).astype(
            np.float32, copy=False
        )

    def decide(self, observation: np.ndarray, *, raw_action=None) -> ActionDecision:
        shared_observation = self.validate_observation(observation)
        base = self.compute_base_action(shared_observation)
        if raw_action is None:
            raw, residual_valid, reason = self._infer_raw_residual(shared_observation)
        else:
            raw = np.asarray(raw_action, dtype=np.float32)
            if raw.shape != ACTION_SHAPE or not np.isfinite(raw).all():
                raise ActionGuardError("external SAC action must be finite and four dimensional")
            residual_valid, reason = True, ""

        if self.enabled and residual_valid:
            mapped = self._map_residual(raw)
            filtered = (
                self.alpha * mapped
                + (1.0 - self.alpha) * self._filtered
            )
            filtered = np.clip(
                filtered, -self.residual_limit, self.residual_limit
            ).astype(np.float32, copy=False)
            self._filtered = filtered.copy()
        else:
            mapped = np.zeros(ACTION_SHAPE, dtype=np.float32)
            filtered = np.zeros(ACTION_SHAPE, dtype=np.float32)
            self.reset_episode()

        final_action = np.clip(
            base + filtered, self.final_low, self.final_high
        ).astype(np.float32, copy=False)
        if final_action.shape != ACTION_SHAPE or not bool(
            np.isfinite(final_action).all()
        ):
            self.reset_episode()
            raise ActionGuardError("fused action failed shape or finite-value validation")
        if bool(
            np.any(final_action < self.final_low)
            or np.any(final_action > self.final_high)
        ):
            self.reset_episode()
            raise ActionGuardError("fused action exceeded final action bounds")

        return ActionDecision(
            base_action=base.copy(),
            raw_residual=raw.copy(),
            mapped_residual=mapped.copy(),
            filtered_residual=filtered.copy(),
            final_action=final_action.copy(),
            residual_enabled=self.enabled,
            residual_valid=residual_valid,
            fallback_reason=reason,
        )


class ResidualGraspEnv:
    """Autonomous wrapper that preserves the complete Day5 execution path."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        controller: ResidualActionController | None = None,
        base_env: Day4GraspEnv | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.controller = controller or ResidualActionController(self.config_path)
        self.base_env = base_env or Day4GraspEnv(self.config_path)
        self.observation: np.ndarray | None = None
        self.last_decision: ActionDecision | None = None

    @property
    def observation_space(self):
        return self.base_env.observation_space

    @property
    def action_space(self):
        return self.base_env.action_space

    def set_residual_enabled(self, enabled: bool) -> None:
        self.controller.set_enabled(enabled)

    def reset(self, **kwargs):
        self.controller.reset_episode()
        self.last_decision = None
        observation, info = self.base_env.reset(**kwargs)
        self.observation = self.controller.validate_observation(observation)
        return self.observation.copy(), info

    def _validate_workspace(self) -> None:
        visual = getattr(self.base_env, "visual", None)
        if not isinstance(visual, dict) or "xyz" not in visual:
            raise ActionGuardError("validated visual target is unavailable")
        xyz = np.asarray(visual["xyz"], dtype=float)
        if xyz.shape != (3,) or not bool(np.isfinite(xyz).all()):
            raise ActionGuardError("visual target is malformed")
        perception = self.controller.config["perception"]
        bounds = (
            perception["workspace_x_m"],
            perception["workspace_y_m"],
            perception["workspace_z_m"],
        )
        if any(
            not float(axis_bounds[0]) <= float(value) <= float(axis_bounds[1])
            for value, axis_bounds in zip(xyz, bounds)
        ):
            raise ActionGuardError("visual target is outside the configured workspace")

    def control_step(self):
        if self.observation is None:
            raise ActionGuardError("reset() must be called before control_step()")
        decision = self.controller.decide(self.observation)
        try:
            self._validate_workspace()
            action = np.asarray(decision.final_action, dtype=np.float32)
            if action.shape != ACTION_SHAPE or not bool(np.isfinite(action).all()):
                raise ActionGuardError("executor input is malformed")
            if bool(
                np.any(action < self.controller.final_low)
                or np.any(action > self.controller.final_high)
            ):
                raise ActionGuardError("executor input exceeds final bounds")
        except ActionGuardError:
            self.controller.reset_episode()
            raise

        observation, reward, terminated, truncated, info = self.base_env.step(action)
        self.observation = self.controller.validate_observation(observation)
        self.last_decision = decision
        merged_info = dict(info)
        merged_info["day6_action"] = decision.as_dict()
        return self.observation.copy(), reward, terminated, truncated, merged_info

    def close(self) -> None:
        self.base_env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--residual", choices=("config", "on", "off"), default="config")
    parser.add_argument("--fixed-center", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    runtime: ResidualGraspEnv | None = None
    try:
        controller = ResidualActionController(config_path)
        if args.residual != "config":
            controller.set_enabled(args.residual == "on")
        runtime = ResidualGraspEnv(config_path, controller=controller)
        for episode in range(args.episodes):
            observation, _ = runtime.reset(
                seed=args.seed + episode,
                options={"fixed_center": args.fixed_center},
            )
            terminated = False
            truncated = False
            steps = 0
            final_info: dict[str, Any] = {}
            while not (terminated or truncated):
                observation, _, terminated, truncated, final_info = (
                    runtime.control_step()
                )
                steps += 1
                print(
                    "DAY6_ACTION="
                    + json.dumps(
                        final_info["day6_action"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    flush=True,
                )
            print(
                f"DAY6_EPISODE={episode + 1} STEPS={steps} "
                f"TERMINATED={int(terminated)} TRUNCATED={int(truncated)} "
                f"SUCCESS={int(bool(final_info.get('success', False)))}",
                flush=True,
            )
        return 0
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    sys.exit(main())
