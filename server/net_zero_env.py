import random
import uuid
from typing import Any

from openenv.core import Environment
from openenv.core.env_server.types import State

try:
    from models import NetZeroAction, NetZeroObservation
except ImportError:
    from ..models import NetZeroAction, NetZeroObservation


class NetZeroEnv(Environment):
    def __init__(self):
        super().__init__()
        self.energy_price = 10.0
        self.humidity = 50.0
        self.carbon_price = 14.0
        self.grid_carbon_intensity = 400.0
        self.renewable_ratio = 0.4
        self.co2_storage_level = 0.0
        self.filter_saturation = 0.0
        self.maintenance_health = 100.0
        self.last_capture_kg = 0.0
        self.last_energy_used_kw = 0.0
        self.last_reward = 0.0
        self.last_efficiency = 1.0
        self.last_action = 0
        self.step_count = 0
        self.max_steps = 24
        self.storage_capacity_kg = 250.0
        self.episode_id = "episode-0"

    @staticmethod
    def _action_value(action: Any) -> int:
        if isinstance(action, dict):
            return int(action.get("action", 0))
        if isinstance(action, NetZeroAction):
            return int(action.action)
        return int(action)

    def _sample_market_and_weather(self) -> None:
        # Simulate mild time-of-day cycles so a policy can exploit timing.
        hour = self.step_count % self.max_steps
        day_peak_multiplier = 1.25 if 12 <= hour <= 19 else 0.9
        wind_factor = random.uniform(0.85, 1.15)
        humidity_shift = random.uniform(-12.0, 12.0)

        self.energy_price = max(5.0, min(20.0, 9.0 * day_peak_multiplier * wind_factor))
        self.humidity = max(5.0, min(95.0, 50.0 + humidity_shift + (6.0 if hour >= 18 else 0.0)))
        self.carbon_price = max(10.0, min(25.0, 14.0 + random.uniform(-2.0, 4.0)))
        self.renewable_ratio = max(0.05, min(0.95, random.uniform(0.2, 0.9)))
        self.grid_carbon_intensity = max(120.0, min(700.0, 700.0 * (1.0 - self.renewable_ratio)))

    def _current_observation(self, message: str) -> NetZeroObservation:
        return NetZeroObservation(
            energy_price=self.energy_price,
            humidity=self.humidity,
            carbon_price=self.carbon_price,
            grid_carbon_intensity=self.grid_carbon_intensity,
            renewable_ratio=self.renewable_ratio,
            co2_storage_level=self.co2_storage_level,
            filter_saturation=self.filter_saturation,
            maintenance_health=self.maintenance_health,
            hour_of_day=self.step_count % self.max_steps,
            message=message,
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> NetZeroObservation:
        self.step_count = 0
        self.episode_id = episode_id or f"episode-{uuid.uuid4()}"
        self.filter_saturation = 0.0
        self.co2_storage_level = 0.0
        self.maintenance_health = 100.0
        self.last_capture_kg = 0.0
        self.last_energy_used_kw = 0.0
        self.last_reward = 0.0
        self.last_efficiency = 1.0
        self.last_action = 0
        self._sample_market_and_weather()
        return self._current_observation("Plant reset. Optimize carbon capture economics over 24 steps.")

    def step(self, action: NetZeroAction, timeout_s: float | None = None, **kwargs) -> NetZeroObservation:
        selected = self._action_value(action)
        base_capture_kg = 0.0
        energy_used_kw = 0.0
        switch_penalty = 0.0
        message = "Idle"

        if selected == 0:
            base_capture_kg = 0.0
            energy_used_kw = 2.0
            message = "Idle mode: no capture."
        elif selected == 1:
            base_capture_kg = 4.0
            energy_used_kw = 14.0
            message = "Low Power: steady capture."
        elif selected == 2:
            base_capture_kg = 10.0
            energy_used_kw = 38.0
            message = "High Power: max capture."
        elif selected == 3:
            self.filter_saturation = 0.0
            base_capture_kg = 0.0
            energy_used_kw = 16.0
            self.maintenance_health = min(100.0, self.maintenance_health + 6.0)
            message = "Filter Purge: saturation reset."
        else:
            obs = self._current_observation("Invalid action. Use 0, 1, 2, or 3.")
            obs.reward = 0.0
            obs.done = True
            return obs

        humidity_efficiency = 1.0
        if self.humidity > 75.0 and selected in (1, 2):
            humidity_efficiency = 0.65 if selected == 2 else 0.8
            base_capture_kg *= humidity_efficiency
            message += " Humidity penalty applied."

        saturation_efficiency = 1.0
        if self.filter_saturation > 90.0 and base_capture_kg > 0.0:
            saturation_efficiency = 0.1
            base_capture_kg *= saturation_efficiency
            message += " Saturation penalty applied."

        health_efficiency = max(0.6, self.maintenance_health / 100.0)
        base_capture_kg *= health_efficiency

        # Capture can be curtailed when storage nears full.
        storage_remaining = max(0.0, self.storage_capacity_kg - self.co2_storage_level)
        captured_kg = min(base_capture_kg, storage_remaining)
        if captured_kg < base_capture_kg:
            message += " Storage cap hit."

        if selected in (1, 2):
            self.filter_saturation = min(100.0, self.filter_saturation + captured_kg * 2.4)

        if selected == 0:
            self.filter_saturation = max(0.0, self.filter_saturation - 1.5)

        if selected == 2 and self.energy_price > 16.0:
            switch_penalty = 4.5
            message += " Peak-price penalty applied."

        changeover_penalty = 0.8 if self.last_action != selected else 0.0
        wear_penalty = 0.4 if selected == 2 else 0.1
        self.maintenance_health = max(45.0, self.maintenance_health - wear_penalty)
        self.co2_storage_level = min(self.storage_capacity_kg, self.co2_storage_level + captured_kg)

        gross_revenue = captured_kg * self.carbon_price
        energy_cost = energy_used_kw * self.energy_price * 0.10
        carbon_cost = (energy_used_kw * self.grid_carbon_intensity / 1000.0) * 0.12
        storage_penalty = 2.0 if self.co2_storage_level > 0.95 * self.storage_capacity_kg else 0.0

        raw_reward = gross_revenue - energy_cost - carbon_cost - switch_penalty - changeover_penalty - storage_penalty
        reward = max(0.0, min(1.0, (raw_reward + 20.0) / 120.0))

        self.last_capture_kg = captured_kg
        self.last_energy_used_kw = energy_used_kw
        self.last_efficiency = humidity_efficiency * saturation_efficiency * health_efficiency
        self.last_reward = reward
        self.last_action = selected

        self.step_count += 1
        done = self.step_count >= self.max_steps

        self._sample_market_and_weather()

        obs = self._current_observation(message)
        obs.reward = reward
        obs.done = done
        return obs

    @property
    def state(self) -> State:
        return State(episode_id=self.episode_id, step_count=self.step_count)
