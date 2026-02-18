# CARLA Lane Keeping — Single Agent PPO

## Overview

This project implements a **reinforcement learning agent** that learns to keep a vehicle within its lane in the [CARLA](https://carla.org/) autonomous driving simulator. The agent is trained end-to-end using **PPO (Proximal Policy Optimization)** from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), with a custom-designed reward function, a stacked-frame visual observation space, and smooth continuous action control.

The work is part of an MSc thesis exploring deep reinforcement learning for autonomous vehicle control.

---

## Demo

> *(Video clip of the agent performing lane keeping — recorded during training)*

<!-- Replace the path below with your actual GIF or video link after uploading to GitHub -->
<!-- ![Agent Demo](assets/demo.gif) -->

📹 A ~30-second clip of the agent executing a smooth lane-keeping maneuver is available in the `assets/` folder.

---

## Key Features

- **Custom Gymnasium environment** wrapping the CARLA simulator
- **PPO** with a multi-layer policy network (256 × 256)
- **Stacked-frame observations** (4 frames × 128×128 RGB) for temporal awareness
- **Continuous action space**: steering, throttle, and brake
- **Shaped reward function** incorporating:
  - Lane center distance penalty
  - Forward progress reward
  - Speed maintenance reward
  - Orientation (yaw error) penalty
  - Collision and lane departure penalties
  - Action smoothness penalties to reduce jerkiness
- **Curriculum learning** support (straight roads → curves → complex scenarios)
- **Linear learning rate schedule** and entropy annealing
- TensorBoard logging and automatic model checkpointing

---

## Project Structure

```
carla-lane-keeping-ppo-single-agent/
│
├── carla_lane_keeping_env.py      # Custom CARLA Gymnasium environment
├── training_lane_keeping_ppo.py   # PPO training script
├── lane_keeping_parameters.py     # All hyperparameters and configuration
├── assets/                        # Demo videos / plots (add manually)
└── README.md
```

---

## Environment Details

| Parameter | Value |
|-----------|-------|
| Simulator | CARLA 0.9.15 |
| Map | Town04 |
| Target Speed | 8.0 m/s (~30 km/h) |
| Max Episode Steps | 2000 |
| Observation | 4 × 128×128 RGB frames + 6D vehicle state vector |
| Action Space | Continuous: `[steering, throttle, brake]` |
| Framework | Stable-Baselines3 (PPO) |

### Reward Function Summary

| Component | Weight |
|-----------|--------|
| Lane center distance | −1.0 |
| Forward progress | +20.0 |
| Speed maintenance | +5.0 |
| Orientation error | −0.5 |
| Collision | −50.0 |
| Lane departure | −20.0 |
| Action smoothness | −0.1 / −0.05 |
| Episode completion bonus | +60.0 |

---

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-4 (linear schedule) |
| Clip Range | 0.2 |
| Entropy Coeff | 0.02 → 0.005 (annealed) |
| GAE Lambda | 0.95 |
| Discount (γ) | 0.99 |
| N Steps | 4096 |
| Batch Size | 512 |
| Epochs per Update | 4 |
| Network | [256, 256] + tanh |
| Total Timesteps | 200,000 |

---

## Requirements

- Windows 10/11 (CARLA runs natively on Windows)
- Python 3.7
- CARLA 0.9.15
- CUDA-capable GPU (recommended)

### Python Dependencies

```bash
pip install stable-baselines3[extra]
pip install gymnasium
pip install numpy
pip install tensorboard
```

---

## Getting Started

### 1. Install CARLA

Download [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15) and extract it. Note the path to the CARLA `.egg` file.

### 2. Configure Paths

Open `lane_keeping_parameters.py` and update these paths to match your system:

```python
BASE_PATH = "your\\path\\to\\project"
CARLA_EGG_PATH = "your\\path\\to\\carla-0.9.15-py3.7-win-amd64.egg"
```

### 3. Launch CARLA Server

Open a terminal and run:

```bash
cd "C:\path\to\CARLA_0.9.15\WindowsNoEditor"
CarlaUE4.exe -quality-level=Low -benchmark -fps=20
```

### 4. Run Training

Open a second terminal and run:

```bash
python training_lane_keeping_ppo.py
```

### 5. Monitor Training (Optional)

```bash
tensorboard --logdir ./logs
```
Then open `http://localhost:6006` in your browser.

---

## Training Notes

- Models are saved every **10,000 timesteps** to the `models/` directory
- Evaluation runs every **5,000 timesteps** over 5 episodes
- Early stopping is triggered if no improvement is seen for **20,000 timesteps**
- Curriculum learning can be enabled via `CurriculumParams.ENABLE_CURRICULUM = True`

---

## Related Project

This is the **single-agent** version of the project. A **multi-agent** extension that trains multiple vehicles simultaneously in the same CARLA world is available here:

👉 [carla-lane-keeping-ppo-multi-agent](https://github.com/YOUR_USERNAME/carla-lane-keeping-ppo-multi-agent)

---

## Citation / Reference

If you use this code in your research, please cite:

```bibtex
@misc{carla_singleagent_ppo,
  author = {Wajdi Bousnina},
  title = {Carla Lane Keeping Single-agent PPO},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/wajdibousnina/carla-lane-keeping-single-agent-ppo/}
}
```

---

## 📧 Contact

**Wajdi Bousnina** - wajdibousnina8@gmail.com

Project Link: [https://github.com/wajdibousnina/carla-lane-keeping-single-agent-ppo/](https://github.com/wajdibousnina/carla-lane-keeping-single-agent-ppo/)
