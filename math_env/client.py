# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Net-Zero Nexus Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import NetZeroAction, NetZeroObservation


class NetZeroEnv(
    EnvClient[NetZeroAction, NetZeroObservation, State]
):
    """
    Client for the Net-Zero Nexus environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with NetZeroEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     result = client.step(NetZeroAction(action=2))
        ...     print(result.observation.message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = NetZeroEnv.from_docker_image("net-zero-nexus:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(NetZeroAction(action=1))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: NetZeroAction) -> Dict:
        """
        Convert NetZeroAction to JSON payload for step message.

        Args:
            action: NetZeroAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[NetZeroObservation]:
        """
        Parse server response into StepResult[NetZeroObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with NetZeroObservation
        """
        obs_data = payload.get("observation", {})
        observation = NetZeroObservation(
            energy_price=obs_data.get("energy_price", 0.0),
            humidity=obs_data.get("humidity", 0.0),
            filter_saturation=obs_data.get("filter_saturation", 0.0),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
