# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Math Env Environment.

The math_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class GuessAction(Action):
    """Action for the Guessing environment - a numeric guess."""

    guess: int = Field(..., description="The number you are guessing")


class GuessObservation(Observation):
    """Observation from the Guessing environment."""

    message: str = Field(default="", description="Feedback message")
