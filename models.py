from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class NetZeroAction(Action):
    action: int = Field(..., description="Plant control action: 0=idle, 1=eco, 2=blast, 3=purge")


class NetZeroObservation(Observation):
    energy_price: float = Field(..., description="Current electricity price (5-20)")
    humidity: float = Field(..., description="Current humidity level (0-100)")
    carbon_price: float = Field(..., description="Current carbon credit price per kg captured")
    grid_carbon_intensity: float = Field(..., description="Grid carbon intensity in gCO2/kWh")
    renewable_ratio: float = Field(..., description="Fraction of renewable energy available (0-1)")
    co2_storage_level: float = Field(..., description="CO2 storage tank fill level in kg")
    filter_saturation: float = Field(..., description="Filter saturation level (0-100)")
    maintenance_health: float = Field(..., description="Plant health score (0-100)")
    hour_of_day: int = Field(..., description="Current hour in the 24-step operating day")
    message: str = Field(default="", description="Environment status message")


# Backward-compatible aliases for earlier naming.
NetZeroActionModel = NetZeroAction
NetZeroObservationModel = NetZeroObservation
