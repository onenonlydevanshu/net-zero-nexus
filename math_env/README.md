---
title: Net-Zero Nexus
emoji: "🚀"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Net-Zero Nexus: Direct Air Capture (DAC) Simulator
**Meta PyTorch OpenEnv Hackathon - Round 1 Submission**

### **Overview**
Net-Zero Nexus is a high-fidelity Reinforcement Learning environment simulating a **Direct Air Capture (DAC)** plant. An AI agent must manage energy consumption, humidity constraints, and filter maintenance to maximize CO2 extraction while minimizing operational costs.

### **The Problem**
Carbon capture is energy-intensive. Running a plant at full power when electricity prices are high or humidity is suboptimal leads to financial loss. This environment challenges agents to find the "Goldilocks" zone of operation.

---

### **Environment Details**

#### **Action Space (NetZeroAction)**
* **0: Idle** - Zero energy use, zero capture.
* **1: Eco Mode** - Lower energy cost with slower but safer capture.
* **2: Blast Mode** - Maximum capture with high energy draw. If electricity price is above 12.0, it auto-shuts down.

#### **Observation Space (NetZeroObservation)**
* `energy_price`: Market cost of electricity (Fluctuates 5.0 - 20.0).
* `humidity`: Ambient air humidity (Impacts capture efficiency).
* `filter_saturation`: Increases with capture. If > 90%, efficiency drops to 10%.
* `carbon_price`: Current carbon-credit price per kg captured.
* `message`: Status message for mode and safety behavior.

#### **Reward Function**
The agent is rewarded based on net economics with wear-and-tear switching penalty:
$$R = (CO_2\_Captured \times CarbonPrice) - (EnergyUsed \times EnergyPrice) - SwitchPenalty$$

---

### **Quick Start**
```python
from server import NetZeroAction, NetZeroEnv

with NetZeroEnv.from_env("devanshu1nonly/net-zero-nexus-v1") as env:
    # Action: 2 (High Power)
    result = env.step(NetZeroAction(action=2))
    print(f"Energy price: {result.observation.energy_price}")
    print(f"Carbon price: {result.observation.carbon_price}")
    print(f"Reward: {result.reward}")
```

### **LLM Config Variables**
The project uses OpenAI-compatible client configuration via environment variables:

```python
from openai import OpenAI
import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - used for local image workflows
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
```

Defaults are set only for `API_BASE_URL` and `MODEL_NAME`.
`HF_TOKEN` has no default and must be provided by the runtime environment.