import json
import os

from models import NetZeroAction
from server import NetZeroEnv


def choose_action(obs) -> int:
    """Heuristic controller that balances economics, reliability, and storage."""
    if obs.filter_saturation > 92.0 or obs.maintenance_health < 55.0:
        return 3

    if obs.co2_storage_level > 235.0:
        return 0

    if obs.energy_price > 16.0 and obs.renewable_ratio < 0.45:
        return 1

    if obs.humidity > 82.0:
        return 1

    if obs.grid_carbon_intensity > 520.0:
        return 1

    return 2


def _serialize_value(value):
    """Convert non-JSON-native values to serializable representations."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _serialize_value(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _serialize_value(value.dict())
    if isinstance(value, dict):
        return str(value)
    if hasattr(value, "__dict__"):
        return str(value.__dict__)
    return str(value)


def log_event(tag: str, payload: dict) -> None:
    """Log event in bracket format: [TAG] key1=value1 key2=value2 ..."""
    try:
        parts = [f"[{tag}]"]
        for key in sorted(payload.keys()):
            val = _serialize_value(payload[key])
            parts.append(f"{key}={val}")
        print(" ".join(parts), flush=True)
    except Exception as exc:  # pragma: no cover - defensive logging fallback
        print(f"[{tag}] error={exc}", flush=True)


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    log_event(
        "START",
        {
            "api_base_url": api_base_url,
            "model_name": model_name,
            "mode": "local_environment_rollout",
        },
    )

    total_reward = 0.0
    env = None

    try:
        env = NetZeroEnv()
        obs = env.reset()

        for step_idx in range(24):
            action_id = choose_action(obs)
            obs = env.step(NetZeroAction(action=action_id))
            step_reward = float(obs.reward or 0.0)
            total_reward += step_reward

            log_event(
                "STEP",
                {
                    "step": step_idx,
                    "action": action_id,
                    "reward": step_reward,
                    "done": bool(obs.done),
                    "energy_price": obs.energy_price,
                    "humidity": obs.humidity,
                    "carbon_price": obs.carbon_price,
                    "renewable_ratio": obs.renewable_ratio,
                    "storage_level": obs.co2_storage_level,
                    "maintenance_health": obs.maintenance_health,
                    "filter_saturation": obs.filter_saturation,
                    "message": obs.message,
                },
            )

            if obs.done:
                break
    except Exception as exc:
        log_event(
            "ERROR",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
    finally:
        log_event(
            "END",
            {
                "total_reward": total_reward,
                "steps_executed": getattr(env, "step_count", 0),
                "final_state": getattr(env, "state", None),
            },
        )


if __name__ == "__main__":
    main()
