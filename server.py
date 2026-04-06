from openenv_core import Action, Observation, StepResult, Environment
from dataclasses import dataclass
import random

@dataclass
class NetZeroAction(Action):
    action: int

@dataclass
class NetZeroObservation(Observation):
    energy_price: float
    humidity: float
    filter_saturation: float
    message: str

class NetZeroEnv(Environment):
    def __init__(self):
        super().__init__()
        self.energy_price = 10.0
        self.humidity = 50.0
        self.filter_saturation = 0.0
        self.last_capture_kg = 0.0
        self.last_energy_used_kw = 0.0
        self.last_reward = 0.0
        self.step_count = 0
        self.max_steps = 24

    def _sample_market_and_weather(self) -> None:
        self.energy_price = float(random.randint(5, 20))
        self.humidity = float(random.randint(0, 100))

    def _current_observation(self, message: str) -> NetZeroObservation:
        return NetZeroObservation(
            energy_price=self.energy_price,
            humidity=self.humidity,
            filter_saturation=self.filter_saturation,
            message=message,
        )

    def reset(self) -> NetZeroObservation:
        self.step_count = 0
        self.filter_saturation = 0.0
        self.last_capture_kg = 0.0
        self.last_energy_used_kw = 0.0
        self.last_reward = 0.0
        self._sample_market_and_weather()
        return self._current_observation("Plant reset. Choose an operating mode.")

    def step(self, action: NetZeroAction) -> StepResult:
        selected = action.action
        base_capture_kg = 0.0
        energy_used_kw = 0.0
        message = "Idle"

        if selected == 0:
            base_capture_kg = 0.0
            energy_used_kw = 0.0
            message = "Idle mode: no capture."
        elif selected == 1:
            base_capture_kg = 2.0
            energy_used_kw = 10.0
            message = "Low Power: steady capture."
        elif selected == 2:
            base_capture_kg = 10.0
            energy_used_kw = 50.0  # 5x Low Power energy usage
            message = "High Power: max capture."
        elif selected == 3:
            self.filter_saturation = 0.0
            base_capture_kg = 0.0
            energy_used_kw = 20.0
            message = "Filter Purge: saturation reset."
        else:
            return StepResult(
                observation=self._current_observation("Invalid action. Use 0, 1, 2, or 3."),
                reward=-10.0,
                done=True,
            )

        # Environmental efficiency: high humidity affects High Power mode.
        if selected == 2 and self.humidity > 75.0:
            base_capture_kg *= 0.5
            message += " Humidity penalty applied."

        # Heavy saturation (>90) drops capture efficiency to 10%.
        if self.filter_saturation > 90.0 and base_capture_kg > 0.0:
            base_capture_kg *= 0.1
            message += " Saturation penalty applied."

        if selected in (1, 2):
            self.filter_saturation = min(100.0, self.filter_saturation + base_capture_kg * 2.0)

        reward = (base_capture_kg * 15.0) - (energy_used_kw * (self.energy_price / 10.0))

        self.last_capture_kg = base_capture_kg
        self.last_energy_used_kw = energy_used_kw
        self.last_reward = reward

        self.step_count += 1
        done = self.step_count >= self.max_steps

        self._sample_market_and_weather()

        return StepResult(
            observation=self._current_observation(message),
            reward=reward,
            done=done,
        )

    def state(self):
        return {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "energy_price": self.energy_price,
            "humidity": self.humidity,
            "filter_saturation": self.filter_saturation,
            "last_capture_kg": self.last_capture_kg,
            "last_energy_used_kw": self.last_energy_used_kw,
            "last_reward": self.last_reward,
        }
