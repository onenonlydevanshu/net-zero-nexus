# 🌱 Net-Zero Nexus

[![Python](https://img.shields.io/badge/Python-71.4%25-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-27.5%25-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License](https://img.shields.io/badge/License-BSD--style-blue?style=flat-square)](./math_env/__init__.py)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](./math_env/Dockerfile)
[![Hackathon](https://img.shields.io/badge/Meta%20PyTorch-OpenEnv%20Hackathon-FF6600?style=flat-square&logo=meta&logoColor=white)](https://github.com/meta-pytorch/OpenEnv)

> A high-fidelity **Reinforcement Learning environment** simulating a **Direct Air Capture (DAC)** plant — where AI agents learn to maximize CO₂ extraction while minimizing operational costs and energy waste.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Examples](#-usage-examples)
- [Environment Details](#-environment-details)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact & Support](#-contact--support)

---

## 🌍 Project Overview

**Net-Zero Nexus** is an open reinforcement learning environment built for the **Meta PyTorch OpenEnv Hackathon (Round 1)**. It simulates a real-world Direct Air Capture (DAC) plant, where carbon dioxide is actively removed from the atmosphere.

### The Challenge

Carbon capture is energy-intensive. Running a plant at full power when electricity prices are high or humidity is suboptimal leads to financial loss and wasted resources. Net-Zero Nexus challenges RL agents to find the *"Goldilocks zone"* of operation — maximizing CO₂ captured per dollar spent while respecting physical constraints like filter wear and humidity effects.

### The Mission

By providing a realistic, economically-grounded simulation of DAC plant operations, Net-Zero Nexus aims to:

- **Accelerate research** into AI-driven climate solutions
- **Lower the barrier** to training and evaluating RL agents for sustainability tasks
- **Bridge the gap** between AI research and real-world carbon capture operations

---

## ✨ Key Features

- 🏭 **Realistic DAC Simulation** — Models energy pricing, humidity impacts, filter saturation, and carbon-credit economics
- 🤖 **RL-Ready Environment** — Compatible with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework; drop-in for standard agent training loops
- 📊 **Structured Logging** — Emits `START`, `STEP`, and `END` events with full episode telemetry
- 🐳 **Docker-First Deployment** — Run the environment server in seconds with a single `docker run` command
- 🔌 **WebSocket Client** — Low-latency persistent connections for efficient multi-step rollouts
- 🌐 **OpenAI-Compatible API** — Plug in any LLM or agent via standard API configuration
- ⚡ **Configurable Market Dynamics** — Randomized energy prices, carbon credit values, and weather per timestep
- 🔁 **24-Step Episode Length** — Simulates a single 24-hour operating day

---

## 🛠 Technology Stack

| Language / Tool | Role |
|---|---|
| **Python 3.10+** | Core environment server, client library, reward modeling, and data models |
| **JavaScript** | Frontend dashboard components and web-based agent interaction |
| **C / C++** | Performance-critical numerical extensions (via compiled dependencies) |
| **Cython** | Python-to-C bindings for compute-intensive simulation paths |
| **CSS** | Styling for the web dashboard |
| **Docker** | Containerized deployment of the environment server |
| **FastAPI + Uvicorn** | HTTP/WebSocket server powering the environment API |
| **Pydantic** | Type-safe action and observation data models |
| **OpenAI SDK** | LLM-based agent integration via OpenAI-compatible API |

---

## 📁 Project Structure

```
net-zero-nexus/
├── math_env/                  # Core DAC environment package
│   ├── __init__.py            # Package exports (NetZeroEnv, NetZeroAction, NetZeroObservation)
│   ├── server.py              # Environment logic, reward function, episode management
│   ├── client.py              # WebSocket client for connecting to the environment server
│   ├── models.py              # Pydantic data models for actions and observations
│   ├── inference.py           # LLM / agent inference utilities
│   ├── Dockerfile             # Container definition for the environment server
│   ├── openenv.yaml           # OpenEnv environment registration manifest
│   ├── pyproject.toml         # Package metadata and dependencies
│   └── README.md              # Environment-specific quick-start guide
├── venv/                      # Python virtual environment (not committed)
├── login_hf.py                # Hugging Face Hub authentication helper
└── README.md                  # ← You are here
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python **3.10** or higher
- [Docker](https://docs.docker.com/get-docker/) (for containerized server)
- A [Hugging Face](https://huggingface.co/) account and write token (for model hub access)

---

### Option 1 — Install as a Python Package

```bash
# Clone the repository
git clone https://github.com/onenonlydevanshu/net-zero-nexus.git
cd net-zero-nexus/math_env

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install the package with all dependencies
pip install -e .
```

---

### Option 2 — Run with Docker

```bash
# Build the image
docker build -t net-zero-nexus:latest ./math_env

# Run the environment server (exposes port 8000)
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  net-zero-nexus:latest
```

---

### Option 3 — Run via Hugging Face Spaces

The environment is also hosted as a Docker Space on Hugging Face Hub:

```python
from math_env import NetZeroEnv

with NetZeroEnv.from_env("devanshu1nonly/net-zero-nexus-v1") as env:
    obs = env.reset()
    print(obs.message)
```

---

### Hugging Face Authentication

```bash
# Using the helper script
python login_hf.py

# Or via the CLI
huggingface-cli login
```

---

## 💡 Usage Examples

### Basic Episode — Manual Control

```python
from math_env import NetZeroEnv, NetZeroAction

# Connect to a running environment server
with NetZeroEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print(f"Initial state: {obs.message}")
    print(f"Energy price: {obs.energy_price}, Humidity: {obs.humidity}")

    # Run a full 24-step episode
    total_reward = 0.0
    done = False
    while not done:
        # Simple heuristic: use Eco mode when energy is cheap, Standby otherwise
        action_id = 1 if obs.energy_price < 10.0 else 0
        result = env.step(NetZeroAction(action=action_id))
        obs = result.observation
        total_reward += result.reward
        done = result.done
        print(f"  Action={action_id} | Reward={result.reward:.2f} | {obs.message}")

    print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
```

---

### Connect to Hugging Face Hosted Environment

```python
from math_env import NetZeroEnv, NetZeroAction

with NetZeroEnv.from_env("devanshu1nonly/net-zero-nexus-v1") as env:
    result = env.step(NetZeroAction(action=2))  # Blast mode
    print(f"Energy price:      {result.observation.energy_price}")
    print(f"Carbon price:      {result.observation.carbon_price}")
    print(f"Filter saturation: {result.observation.filter_saturation:.1f}%")
    print(f"Reward:            {result.reward:.2f}")
```

---

### LLM-Based Agent

```python
import os
from openai import OpenAI
from math_env import NetZeroEnv, NetZeroAction

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("HF_TOKEN"),
)

with NetZeroEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()

    for step in range(24):
        prompt = (
            f"You are controlling a Direct Air Capture plant. "
            f"Energy price: {obs.energy_price}, Humidity: {obs.humidity}, "
            f"Filter saturation: {obs.filter_saturation:.1f}%, "
            f"Carbon price: {obs.carbon_price}. "
            f"Choose action 0 (Standby), 1 (Eco), or 2 (Blast). Reply with only the number."
        )
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        action_id = int(response.choices[0].message.content.strip())
        result = env.step(NetZeroAction(action=action_id))
        obs = result.observation
        if result.done:
            break
```

---

## 🌐 Environment Details

### Action Space

| Action ID | Mode | Energy Use | CO₂ Capture | Notes |
|---|---|---|---|---|
| `0` | **Standby** | 1.0 kW | 0 kg | Idles the plant; filter slowly regenerates |
| `1` | **Eco Mode** | 12.0 kW | ~4.0 kg | Steady, safe, moderate capture |
| `2` | **Blast Mode** | 35.0 kW | ~10.0 kg | Maximum capture; **auto-shuts down** if energy price > 12.0 |

### Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `energy_price` | `float` | 5.0 – 20.0 | Current market electricity cost |
| `humidity` | `float` | 0 – 100 | Ambient air humidity (affects capture efficiency) |
| `filter_saturation` | `float` | 0 – 100 | Filter fill level; efficiency drops above 90% |
| `carbon_price` | `float` | 10.0 – 20.0 | Carbon credit value per kg CO₂ captured |
| `message` | `str` | — | Human-readable status/event description |

### Reward Function

$$R = (CO_2\text{\_captured} \times \text{carbon\_price}) - (\text{energy\_used} \times \text{energy\_price}) - \text{switch\_penalty}$$

- **Switch penalty:** `1.2` if the selected action differs from the previous step (discourages rapid mode switching)
- **Episode length:** 24 steps (one simulated operating day)

### Efficiency Modifiers

- **Humidity** > 70% progressively reduces capture efficiency (down to 35% minimum)
- **Filter saturation** progressively reduces capture efficiency (based on `max(0.25, 1 - saturation/140)`)

---

## ⚙️ Configuration

The server reads configuration from environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model name for agent inference |
| `HF_TOKEN` | *(required)* | Hugging Face API token |
| `LOCAL_IMAGE_NAME` | *(optional)* | Local Docker image name for offline workflows |

---

## 🤝 Contributing

Contributions are warmly welcomed! Whether it's a bug fix, a new agent strategy, improved documentation, or an enhanced reward function — all help is appreciated.

### Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally (replace `YOUR_USERNAME` with your GitHub username):
   ```bash
   git clone https://github.com/YOUR_USERNAME/net-zero-nexus.git
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dev dependencies:**
   ```bash
   pip install -e "math_env[dev]"
   ```
5. **Make your changes** and add tests where applicable
6. **Run tests:**
   ```bash
   pytest math_env/
   ```
7. **Open a Pull Request** with a clear description of your changes

### Contribution Ideas

- 🧪 Additional RL agent baselines (PPO, DQN, SAC)
- 📈 Visualization tools for episode telemetry
- 🌡️ More realistic weather and energy price simulation
- 🔬 Extended environment variants (e.g., multi-plant, grid-tied storage)
- 📝 Tutorials and notebooks for newcomers

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Use type hints throughout
- Add docstrings to all public classes and functions

---

## 📄 License

This project is licensed under the **BSD-style license** used by Meta Platforms, Inc. and its affiliates.

> Copyright (c) Meta Platforms, Inc. and affiliates.  
> All rights reserved.  
> See the `LICENSE` file in the root of this source tree for details.

---

## 📬 Contact & Support

| Channel | Link |
|---|---|
| **GitHub Issues** | [Open an issue](https://github.com/onenonlydevanshu/net-zero-nexus/issues) |
| **GitHub Profile** | [@onenonlydevanshu](https://github.com/onenonlydevanshu) |
| **Hugging Face** | [devanshu1nonly](https://huggingface.co/devanshu1nonly) |

---

<div align="center">
  <sub>Built with ❤️ for a net-zero future · Meta PyTorch OpenEnv Hackathon 2025</sub>
</div>
