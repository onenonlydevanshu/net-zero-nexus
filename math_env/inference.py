import asyncio
import json
import os

from openai import OpenAI

try:
    from .client import NetZeroEnv
    from .models import NetZeroAction
except ImportError:
    from client import NetZeroEnv
    from models import NetZeroAction


# Defaults are intentionally set only for API_BASE_URL and MODEL_NAME.
API_BASE_URL = os.getenv("API_BASE_URL", "https://devanshu1nonly-net-zero-nexus.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional when using NetZeroEnv.from_docker_image(...)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def log_event(tag: str, payload: dict) -> None:
    # Required format: START / STEP / END
    print(f"{tag} {json.dumps(payload, separators=(',', ':'))}")


def choose_action(llm_client: OpenAI, obs, step_idx: int) -> int:
    # All LLM calls are routed through this OpenAI client configured by env vars.
    if not HF_TOKEN:
        if obs.energy_price > 12.0:
            return 0
        if obs.humidity > 82.0:
            return 1
        return 2

    prompt = (
        "You are controlling a DAC plant. Return exactly one integer in {0,1,2}. "
        f"Step={step_idx}, energy_price={obs.energy_price}, humidity={obs.humidity}, "
        f"filter_saturation={obs.filter_saturation}, carbon_price={obs.carbon_price}."
    )

    try:
        response = llm_client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            max_output_tokens=8,
        )
        text = (response.output_text or "").strip()
        for ch in text:
            if ch in "012":
                return int(ch)
    except Exception:
        pass

    # Safe fallback when model output is malformed or API call fails.
    if obs.energy_price > 12.0:
        return 0
    if obs.humidity > 82.0:
        return 1
    return 2


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL.rstrip("/") + "/v1", api_key=HF_TOKEN)

    conn_meta = {
        "api_base_url": API_BASE_URL,
        "model_name": MODEL_NAME,
        "uses_local_image": bool(LOCAL_IMAGE_NAME),
    }
    log_event("START", conn_meta)

    if LOCAL_IMAGE_NAME:
        env = await NetZeroEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = NetZeroEnv(base_url=API_BASE_URL)

    total_reward = 0.0
    max_steps = 10
    final_done = False

    try:
        reset_result = await env.reset()
        obs = reset_result.observation

        for step_idx in range(max_steps):
            action_id = choose_action(llm_client, obs, step_idx)
            step_result = await env.step(NetZeroAction(action=action_id))
            obs = step_result.observation
            total_reward += float(step_result.reward or 0.0)

            log_event(
                "STEP",
                {
                    "index": step_idx,
                    "action": action_id,
                    "reward": step_result.reward,
                    "done": step_result.done,
                    "energy_price": obs.energy_price,
                    "humidity": obs.humidity,
                    "filter_saturation": obs.filter_saturation,
                },
            )

            if step_result.done:
                final_done = True
                break

        log_event("END", {"total_reward": total_reward, "final_done": final_done})
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
