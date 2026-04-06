from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class NetZeroActionModel(Action):
    action: int = Field(..., description="Plant control action: 0, 1, 2, or 3")


class NetZeroObservationModel(Observation):
    energy_price: float = Field(..., description="Current electricity price (5-20)")
    humidity: float = Field(..., description="Current humidity level (0-100)")
    filter_saturation: float = Field(..., description="Filter saturation level (0-100)")
    message: str = Field(default="", description="Environment status message")
