# Installation & Setup Guide

Complete setup instructions for CARLA Lane Keeping PPO project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [CARLA Installation](#carla-installation)
3. [Python Environment Setup](#python-environment-setup)
4. [Project Configuration](#project-configuration)
5. [Verification](#verification)
6. [Parameter Tuning Guide](#parameter-tuning-guide)
7. [Common Issues](#common-issues)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+
- **CPU**: Quad-core 2.5 GHz+
- **RAM**: 8 GB
- **GPU**: NVIDIA GPU with 4GB VRAM (GTX 1060 or better)
- **Storage**: 20 GB free space
- **Python**: 3.7 (exactly - CARLA 0.9.15 requirement)

### Recommended
- **CPU**: 6-core 3.0 GHz+ (e.g., i5-12400F)
- **RAM**: 16 GB
- **GPU**: RTX 3060 (12GB) or better
- **Storage**: 50 GB free space (for training logs)

---

## CARLA Installation

### Step 1: Download CARLA

1. Go to [CARLA Releases](https://github.com/carla-simulator/carla/releases)
2. Download **CARLA 0.9.15** for your OS:
   - Windows: `CARLA_0.9.15.zip`
   - Linux: `CARLA_0.9.15.tar.gz`

3. Extract to a permanent location:
   ```
   Windows: C:\Program Files\CARLA_0.9.15\
   Linux: /opt/CARLA_0.9.15/
   ```

### Step 2: Verify CARLA Works

**Windows:**
```bash
cd C:\Program Files\CARLA_0.9.15\WindowsNoEditor
CarlaUE4.exe
```

**Linux:**
```bash
cd /opt/CARLA_0.9.15
./CarlaUE4.sh
```

You should see the CARLA window open. Press `ESC` to quit.

### Step 3: Find Python API Egg File

The egg file is located at:
- **Windows**: `CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg`
- **Linux**: `CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg`

**Save this path** — you'll need it for configuration.

---

## Python Environment Setup

### Step 1: Install Python 3.7

**Critical**: CARLA 0.9.15 requires **exactly Python 3.7**

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/release/python-3710/)
2. Install with "Add to PATH" checked
3. Verify: `python --version` → should show `Python 3.7.x`

**Linux:**
```bash
sudo apt update
sudo apt install python3.7 python3.7-dev python3.7-venv
```

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd carla-lane-keeping-single-agent-ppo

# Create virtual environment
python3.7 -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r docs/requirements.txt
```

**Expected installation time**: 5–10 minutes

### Step 4: Install PyTorch with CUDA (if using GPU)

**For CUDA 11.7** (RTX 30-series):
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

**For CUDA 11.6**:
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

**CPU only** (not recommended):
```bash
pip install torch==1.13.1 torchvision==0.14.1
```

Verify CUDA:
```python
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## Project Configuration

### Step 1: Clone Repository

```bash
git clone https://github.com/wajdibousnina/carla-lane-keeping-single-agent-ppo.git
cd carla-lane-keeping-single-agent-ppo
```

### Step 2: Configure CARLA Path

Open `main_files/lane_keeping_parameters.py` and update:

```python
CARLA_EGG_PATH = "C:\\Program Files\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg"

# Or for Linux:
CARLA_EGG_PATH = "/opt/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
```

> **Important**: Use double backslashes `\\` on Windows!

### Step 3: Configure Training Directory

```python
BASE_PATH = "C:\\your\\preferred\\output\\folder"   # Windows
# Or:
BASE_PATH = "/home/username/carla_training"          # Linux
```

This folder will store trained models, TensorBoard logs, and plots.

---

## Verification

### Test 1: CARLA Connection

Start CARLA, then in a Python prompt:

```python
import sys
sys.path.append("YOUR_CARLA_EGG_PATH")
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print("✓ Connected to CARLA")
```

### Test 2: Environment Creation

```python
from main_files.carla_lane_keeping_env import CarlaLaneKeepingEnv
env = CarlaLaneKeepingEnv()
# Should print: "✓ CARLA connected successfully"
env.close()
```

### Test 3: Quick Training Run

```bash
python main_files/training_lane_keeping_ppo.py
```

Reduce `TOTAL_TIMESTEPS` to `5000` in the parameters file for a quick sanity check.

If all tests pass: **✓ Setup complete!**

---

## Parameter Tuning Guide

All training behavior is controlled from a single file: **`main_files/lane_keeping_parameters.py`**. This section explains every configurable section and what to change depending on your goals.

---

### 1. Paths (top of file)

```python
BASE_PATH = "F:\\thesis_carla_lane_keeping"
CARLA_EGG_PATH = "F:\\Program files\\CARLA_0.9.15\\..."
```

These two lines **must** be updated to match your system before anything else will work. `BASE_PATH` is where all output files (models, logs, plots) will be saved.

---

### 2. Environment Parameters

```python
MAX_EPISODE_STEPS = 2000
TARGET_SPEED = 8.0       # m/s ≈ 30 km/h
MIN_SPEED_THRESHOLD = 1.0
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `MAX_EPISODE_STEPS` | Maximum steps before episode is cut off | Increase if agent needs more time to travel; decrease to speed up training cycles |
| `TARGET_SPEED` | The speed the agent is rewarded for maintaining | Increase for highway-like scenarios; decrease for tight/curved roads |
| `MIN_SPEED_THRESHOLD` | Agent is penalized if speed falls below this | Prevents the agent from standing still to avoid collisions; raise if agent is lazy |
| `SPAWN_POINT_INDEX` | `None` = random spawn; integer = fixed spawn | Use fixed index during debugging so the scenario is reproducible |
| `WEATHER_RANDOMIZATION` | Randomize weather each episode | Enable later in training to improve generalization |

---

### 3. Reward Function — `RewardParams`

This is the most important section for shaping agent behavior. Every value here directly changes what the agent "wants" to do.

```python
class RewardParams:
    LANE_CENTER_WEIGHT = -1.0
    FORWARD_PROGRESS_WEIGHT = 20.0
    ORIENTATION_WEIGHT = -0.5
    SPEED_REWARD_WEIGHT = 5.0
    MIN_SPEED_PENALTY = -5.0
    COLLISION_PENALTY = -50.0
    LANE_DEPARTURE_PENALTY = -20.0
    REVERSE_PENALTY = -5.0
    STEERING_SMOOTHNESS_WEIGHT = -0.1
    THROTTLE_SMOOTHNESS_WEIGHT = -0.05
    COMPLETION_BONUS = 60.0
```

**How to read these values:** positive = reward (agent wants more of it), negative = penalty (agent tries to avoid it). Larger magnitude = stronger signal.

| Parameter | Effect | Tuning Tip |
|-----------|--------|-----------|
| `LANE_CENTER_WEIGHT` | Penalizes distance from lane center | Make more negative (e.g., `-2.0`) if agent drifts too far; careful not to make it so large the agent stops moving |
| `FORWARD_PROGRESS_WEIGHT` | Rewards covering distance forward | The main driving force — if agent is lazy/stationary, increase this |
| `ORIENTATION_WEIGHT` | Penalizes misalignment with road direction | Increase magnitude if agent drives at an angle; too high can cause oscillation |
| `SPEED_REWARD_WEIGHT` | Rewards driving close to `TARGET_SPEED` | Reduce if agent overspeeds dangerously; increase if agent drives too slowly |
| `MIN_SPEED_PENALTY` | Penalizes being nearly stopped | Increase magnitude if agent stops frequently to "avoid" other penalties |
| `COLLISION_PENALTY` | Large one-time penalty when a collision occurs | Keep large relative to other rewards so the agent treats collision as a hard failure |
| `LANE_DEPARTURE_PENALTY` | Penalty when vehicle fully exits the lane | Should be large enough to discourage lane crossing but less than collision |
| `STEERING_SMOOTHNESS_WEIGHT` | Penalizes abrupt steering changes | Increase magnitude if agent steers jerkily; but too large may prevent sharp-turn response |
| `COMPLETION_BONUS` | Bonus for finishing the episode without failure | Helps the agent learn long-horizon behavior; reduce if agent only cares about the bonus |

**Common behavior problems and reward fixes:**

- **Agent stops or slows down too much** → Increase `FORWARD_PROGRESS_WEIGHT` and `MIN_SPEED_PENALTY` magnitude
- **Agent drives in a zigzag** → Increase `ORIENTATION_WEIGHT` and `STEERING_SMOOTHNESS_WEIGHT` magnitude
- **Agent ignores lane boundaries** → Increase `LANE_CENTER_WEIGHT` and `LANE_DEPARTURE_PENALTY` magnitude
- **Agent drives recklessly fast** → Reduce `SPEED_REWARD_WEIGHT` or reduce `TARGET_SPEED`

---

### 4. Observation Space — `ObservationParams`

```python
class ObservationParams:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    FRAME_STACK = 4
    SPEED_NORMALIZATION = 15.0
    DISTANCE_NORMALIZATION = 4.0
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `IMAGE_HEIGHT` / `IMAGE_WIDTH` | Resolution of camera frames fed to the network | Reduce to `96×96` if running out of GPU VRAM; increase for more visual detail (much slower) |
| `FRAME_STACK` | How many consecutive frames are stacked as one observation | 4 is standard; reduce to 3 to save memory; increase for faster dynamics |
| `SPEED_NORMALIZATION` | Divides raw speed to normalize to ~[0,1] range | Set to roughly the max speed you expect the vehicle to reach |
| `DISTANCE_NORMALIZATION` | Divides lane offset distance to normalize | Set to roughly the lane width (4m is typical for CARLA) |

> **Note:** If you change `IMAGE_HEIGHT`, `IMAGE_WIDTH`, or `FRAME_STACK`, the model architecture changes and any saved checkpoint becomes incompatible. Always start fresh training after changing these.

---

### 5. Action Space — `ActionParams`

```python
class ActionParams:
    STEERING_SMOOTHING = 0.3
    THROTTLE_SMOOTHING = 0.3
    BRAKE_SMOOTHING = 0.3
    MAX_STEERING_CHANGE = 0.5
    MAX_THROTTLE_CHANGE = 0.8
    MAX_BRAKE_CHANGE = 0.8
```

Action smoothing blends the new action with the previous one to avoid sudden jerks. A value of `0.3` means: `actual_action = 0.3 × new + 0.7 × previous`.

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `STEERING_SMOOTHING` | How quickly steering responds to policy output | Increase (towards 1.0) for more responsive steering; decrease for smoother but slower response |
| `THROTTLE_SMOOTHING` | Same for throttle | Reduce if acceleration feels laggy |
| `MAX_STEERING_CHANGE` | Maximum steering change allowed per step | Reduce to prevent sudden swerves; increase for tight turns |
| `MAX_THROTTLE_CHANGE` | Maximum throttle change per step | Reduce for smoother acceleration profile |

---

### 6. PPO Hyperparameters — `PPOParams`

```python
class PPOParams:
    LEARNING_RATE = 5e-4
    LR_SCHEDULE = 'linear'
    CLIP_RANGE = 0.2
    ENTROPY_COEF = 0.02
    ENTROPY_COEF_FINAL = 0.005
    N_STEPS = 4096
    BATCH_SIZE = 512
    N_EPOCHS = 4
    GAE_LAMBDA = 0.95
    GAMMA = 0.99
    NET_ARCH = [256, 256]
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `LEARNING_RATE` | How fast the network weights update | Lower (e.g., `1e-4`) if training is unstable; raise (e.g., `1e-3`) if learning is too slow |
| `LR_SCHEDULE` | `'linear'` decays LR to 0 over training; `'constant'` keeps it fixed | Use `'linear'` for most cases; `'constant'` for fine-tuning a loaded checkpoint |
| `CLIP_RANGE` | Limits how much the policy changes per update (PPO core) | Keep at `0.2`; lower to `0.1` for more conservative updates |
| `ENTROPY_COEF` | Initial exploration bonus — encourages trying diverse actions | Increase if agent gets stuck in repetitive behavior early on |
| `ENTROPY_COEF_FINAL` | Entropy after annealing — controls late-stage exploration | Keep low; agent should exploit learned policy by end of training |
| `N_STEPS` | Steps collected from the environment before each update | Increase for more stable gradient estimates (uses more memory) |
| `BATCH_SIZE` | Mini-batch size during the update step | Reduce to `256` if getting CUDA out-of-memory errors |
| `N_EPOCHS` | How many times each collected batch is reused per update | Keep at 4; increase to 8–10 for sample efficiency at the cost of stability |
| `GAMMA` | Discount factor — how much the agent values future rewards | Close to 1.0 (e.g., 0.99) for long-horizon tasks; lower for short-term focus |
| `GAE_LAMBDA` | Advantage estimation smoothing | Higher = lower variance but more bias; `0.95` is a good default |
| `NET_ARCH` | Hidden layer sizes of the policy network | Increase to `[512, 512]` for more complex scenarios; ensure VRAM allows it |

---

### 7. Training Parameters — `TrainingParams`

```python
class TrainingParams:
    TOTAL_TIMESTEPS = 200_000
    SAVE_FREQ = 10_000
    EVAL_FREQ = 5_000
    N_EVAL_EPISODES = 5
    PATIENCE = 20_000
```

| Parameter | What it does | When to change it |
|-----------|-------------|-------------------|
| `TOTAL_TIMESTEPS` | Total environment steps before training ends | Increase significantly (500k–1M) for harder tasks or better final performance |
| `SAVE_FREQ` | How often (in steps) a model checkpoint is saved | Decrease if you want finer recovery options; increase to save disk space |
| `EVAL_FREQ` | How often the model is evaluated on test episodes | Decrease for more frequent feedback; increase to speed up training |
| `N_EVAL_EPISODES` | Number of episodes used in each evaluation | Increase for a more reliable average (but evaluation is slow) |
| `PATIENCE` | Early stopping: stops if no improvement after this many steps | Increase if training needs more time to improve; decrease to stop faster |

---

### 8. Curriculum Learning — `CurriculumParams`

```python
class CurriculumParams:
    ENABLE_CURRICULUM = True
    STAGE_1_EPISODES = 200    # Straight roads, no traffic
    STAGE_2_EPISODES = 300    # Slight curves, some weather
    STAGE_3_EPISODES = inf    # Full complexity
```

Curriculum learning starts training in easy conditions and progressively increases difficulty. This generally leads to faster and more stable learning than starting with full complexity.

To **disable** curriculum and train on full complexity from the start:
```python
ENABLE_CURRICULUM = False
```

To adjust stage length, change `STAGE_1_EPISODES` and `STAGE_2_EPISODES`. If the agent is struggling on Stage 1, increase `STAGE_1_EPISODES` to give it more time on straight roads before moving on.

---

## Common Issues

### Issue 1: "ImportError: No module named 'carla'"

**Cause**: CARLA egg path not configured correctly.

**Fix**: Update `CARLA_EGG_PATH` in `lane_keeping_parameters.py` with the full absolute path, using double backslashes on Windows.

### Issue 2: "Connection refused" when connecting to CARLA

**Cause**: CARLA server is not running.

**Fix**: Start `CarlaUE4.exe`, wait 15–20 seconds, then retry.

### Issue 3: "CUDA out of memory"

**Fix**:
```python
class PPOParams:
    BATCH_SIZE = 256        # Reduce from 512

class ObservationParams:
    IMAGE_HEIGHT = 96       # Reduce from 128
    IMAGE_WIDTH = 96
```

### Issue 4: Python version mismatch

CARLA 0.9.15 requires **exactly Python 3.7**. You must create a venv using Python 3.7; newer versions are not compatible.

### Issue 5: Training very slow

1. Verify GPU is in use: `python -c "import torch; print(torch.cuda.is_available())"`
2. Launch CARLA in low resolution: `CarlaUE4.exe -ResX=640 -ResY=480 -quality-level=Low`
3. Close other GPU-heavy applications

### Issue 6: "Vehicle spawn collision"

```python
SPAWN_POINT_INDEX = None   # Random spawn (default)
# Or try a specific index if you know one is free:
SPAWN_POINT_INDEX = 15
```

### Issue 7: Training crashes after many steps

**Cause**: RAM exhaustion or CARLA server instability.

**Fix**: Reduce `FRAME_STACK` from 4 to 3, and consider restarting the CARLA server every 100k steps for long training runs.

---

## Next Steps

After successful setup:

1. **Quick test**: Run with `TOTAL_TIMESTEPS = 5_000` to verify everything works end-to-end
2. **Monitor training**: Open TensorBoard to watch reward curves
3. **Tune reward weights**: Adjust `RewardParams` based on the behavior you observe
4. **Scale up**: Increase to full 200k+ timesteps once the reward trends look healthy

---

## Support

If you encounter issues not covered here, open an issue on the [GitHub repository](https://github.com/wajdibousnina/carla-lane-keeping-single-agent-ppo/issues) with:
- The full error message
- Your system specs (OS, GPU, RAM)
- Steps to reproduce the problem

