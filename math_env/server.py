from openenv_core import Environment
try:
    from openenv_core import StepResult
except ImportError:
    from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import NetZeroAction, NetZeroObservation
from openai import OpenAI
import json
import os
import random
from uuid import uuid4


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional for local image-based workflows.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def get_openai_client() -> OpenAI:
    """Create the OpenAI client using environment-based configuration."""
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _log_event(tag: str, payload: dict) -> None:
    """Emit required structured logs with START/STEP/END tags."""
    print(f"{tag} {json.dumps(payload, sort_keys=True)}", flush=True)

class NetZeroEnv(Environment):
    def __init__(self):
        super().__init__()
        self.energy_price = 10.0
        self.humidity = 50.0
        self.carbon_price = 12.0
        self.filter_saturation = 0.0
        self.last_capture_kg = 0.0
        self.last_energy_used_kw = 0.0
        self.last_reward = 0.0
        self.last_wear_penalty = 0.0
        self.last_action = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 24  # A single 24-hour operating day.

    def _sample_market_and_weather(self) -> None:
        self.energy_price = float(random.randint(5, 15))
        self.humidity = float(random.randint(0, 100))
        self.carbon_price = round(random.uniform(10.0, 20.0), 2)

    def _capture_efficiency(self) -> float:
        # Humidity and saturation both reduce capture efficiency.
        humidity_eff = 1.0 if self.humidity <= 70.0 else max(0.35, 1.0 - ((self.humidity - 70.0) / 60.0))
        saturation_eff = max(0.25, 1.0 - (self.filter_saturation / 140.0))
        return humidity_eff * saturation_eff

    def _current_observation(self, message: str) -> NetZeroObservation:
        return NetZeroObservation(
            energy_price=self.energy_price,
            humidity=self.humidity,
            filter_saturation=self.filter_saturation,
            carbon_price=self.carbon_price,
            message=message,
        )

    def reset(self) -> NetZeroObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.filter_saturation = 0.0
        self.last_capture_kg = 0.0
        self.last_energy_used_kw = 0.0
        self.last_reward = 0.0
        self.last_wear_penalty = 0.0
        self.last_action = 0
        self._sample_market_and_weather()
        _log_event(
            "START",
            {
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "energy_price": self.energy_price,
                "humidity": self.humidity,
                "filter_saturation": self.filter_saturation,
                "carbon_price": self.carbon_price,
            },
        )
        return self._current_observation("Plant reset. Choose an operating mode.")

    def step(self, action: NetZeroAction) -> StepResult:
        selected = action.action
        base_capture_kg = 0.0
        energy_used_kw = 0.0
        message = "Idle"

        if selected == 0:
            base_capture_kg = 0.0
            energy_used_kw = 1.0
            message = "Standby mode: waiting for better conditions."
        elif selected == 1:
            base_capture_kg = 4.0
            energy_used_kw = 12.0
            message = "Eco mode: safe, slower capture."
        elif selected == 2:
            base_capture_kg = 10.0
            energy_used_kw = 35.0
            message = "Blast mode: maximum capture."

            if self.energy_price > 12.0:
                base_capture_kg = 0.0
                energy_used_kw = 2.0
                message = "Blast mode auto-shutdown: electricity too expensive."
        else:
            _log_event(
                "STEP",
                {
                    "episode_id": self._state.episode_id,
                    "step_count": self._state.step_count,
                    "action": selected,
                    "reward": -10.0,
                    "done": True,
                },
            )
            _log_event(
                "END",
                {
                    "episode_id": self._state.episode_id,
                    "step_count": self._state.step_count,
                    "reason": "invalid_action",
                },
            )
            return StepResult(
                observation=self._current_observation("Invalid action. Use 0, 1, or 2."),
                reward=-10.0,
                done=True,
            )

        efficiency = self._capture_efficiency()
        captured_kg = base_capture_kg * efficiency

        if selected in (1, 2):
            self.filter_saturation = min(100.0, self.filter_saturation + captured_kg * 2.2)
        else:
            self.filter_saturation = max(0.0, self.filter_saturation - 4.0)

        switch_penalty = 1.2 if selected != self.last_action else 0.0
        reward = (captured_kg * self.carbon_price) - (energy_used_kw * self.energy_price) - switch_penalty

        self.last_capture_kg = captured_kg
        self.last_energy_used_kw = energy_used_kw
        self.last_reward = reward
        self.last_wear_penalty = switch_penalty
        self.last_action = selected

        self._state.step_count += 1
        done = self._state.step_count >= self.max_steps

        self._sample_market_and_weather()

        _log_event(
            "STEP",
            {
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "action": selected,
                "captured_kg": captured_kg,
                "energy_used_kw": energy_used_kw,
                "wear_penalty": switch_penalty,
                "reward": reward,
                "done": done,
            },
        )
        if done:
            _log_event(
                "END",
                {
                    "episode_id": self._state.episode_id,
                    "step_count": self._state.step_count,
                    "reason": "max_steps",
                },
            )

        return StepResult(
            observation=self._current_observation(message),
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state
