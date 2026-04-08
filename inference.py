import json
import os

from openai import OpenAI
from models import NetZeroAction
from server import NetZeroEnv


def get_llm_action(llm_client: OpenAI, obs, step_idx: int) -> int:
    """Query LLM for action recommendation via injected API_BASE_URL."""
    try:
        prompt = (
            f"Control a DAC plant to maximize carbon capture over economics. "
            f"Step {step_idx}/24. Current state: energy_price={obs.energy_price:.1f}, "
            f"humidity={obs.humidity:.1f}, filter_saturation={obs.filter_saturation:.1f}. "
            f"Return exactly one digit (0=idle, 1=eco, 2=blast, 3=purge)."
        )
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        for ch in text:
            if ch in "0123":
                return int(ch)
    except Exception:
        pass
    return None


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
    api_key = os.getenv("API_KEY", "test-key")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

    log_event(
        "START",
        {
            "api_base_url": api_base_url,
            "model_name": model_name,
            "mode": "llm_rollout",
        },
    )

    total_reward = 0.0
    env = None
    llm_client = None

    try:
        # Initialize OpenAI client with injected environment variables
        llm_client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )

        env = NetZeroEnv()
        obs = env.reset()

        for step_idx in range(24):
            # Try LLM action first, fall back to heuristic if it fails
            action_id = get_llm_action(llm_client, obs, step_idx)
            if action_id is None:
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
