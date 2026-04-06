# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Math Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import GuessAction, GuessObservation
except ImportError:
    from models import GuessAction, GuessObservation


class GuessingEnv(Environment):
    """
    A simple number guessing environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Guessing environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.secret_number = 0

    def reset(self) -> GuessObservation:
        """
        Reset the environment and pick a secret number.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.secret_number = random.randint(1, 100)

        return GuessObservation(
            message="Environment Reset. Guess a number between 1 and 100.",
            done=False,
            reward=0.0,
        )

    def step(self, action: GuessAction) -> GuessObservation:  # type: ignore[override]
        """
        Execute a step in the environment.
        """
        self._state.step_count += 1

        if action.guess == self.secret_number:
            reward = 1.0
            done = True
            msg = "Correct Guess!"
        else:
            reward = 0.0
            done = False
            msg = "Wrong. Try again."

        return GuessObservation(
            message=msg,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
