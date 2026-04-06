from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class NetZeroAction(Action):
    action: int = Field(..., description="Plant control action: 0=Standby, 1=Eco, 2=Blast")


class NetZeroObservation(Observation):
    energy_price: float = Field(..., description="Current electricity price (5-20)")
    humidity: float = Field(..., description="Current humidity level (0-100)")
    filter_saturation: float = Field(..., description="Filter saturation level (0-100)")
    carbon_price: float = Field(..., description="Current carbon credit value per kg CO2")
    message: str = Field(default="", description="Environment status message")
