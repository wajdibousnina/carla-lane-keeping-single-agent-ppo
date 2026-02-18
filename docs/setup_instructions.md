# Installation & Setup Guide

Complete setup instructions for CARLA Lane Keeping PPO project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [CARLA Installation](#carla-installation)
3. [Python Environment Setup](#python-environment-setup)
4. [Project Configuration](#project-configuration)
5. [Verification](#verification)
6. [Common Issues](#common-issues)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+
- **CPU**: Quad-core 2.5 GHz+
- **RAM**: 8 GB
- **GPU**: NVIDIA GPU with 4GB VRAM (GTX 1060 or better)
- **Storage**: 20 GB free space
- **Python**: 3.7 (exactly - CARLA 0.9.15 requirement)

### Recommended for Multi-Agent
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

**Save this path** - you'll need it for configuration.

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
cd carla-lane-keeping-ppo

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
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes

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
git clone https://github.com/YOUR_USERNAME/carla-lane-keeping-ppo.git
cd carla-lane-keeping-ppo
```

### Step 2: Configure CARLA Path

Edit `src/lane_keeping_parameters.py`:

```python
# Line ~10: Update with YOUR path
CARLA_EGG_PATH = "C:\\Program Files\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg"

# Or for Linux:
CARLA_EGG_PATH = "/opt/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
```

**Important**: Use double backslashes `\\` on Windows!

### Step 3: Configure Training Directory

Edit `src/lane_keeping_parameters.py`:

```python
# Line ~8: Update base path
BASE_PATH = "F:\\thesis_carla_lane_keeping"  # Windows
# Or:
BASE_PATH = "/home/username/carla_training"  # Linux
```

This folder will store:
- Trained models
- Training logs
- TensorBoard data
- Plots

### Step 4: Adjust Parameters (Optional)

For quick testing, reduce training time:

```python
# In lane_keeping_parameters.py
class TrainingParams:
    TOTAL_TIMESTEPS = 50_000  # Down from 200k for quick test
```

---

## Verification

### Test 1: CARLA Connection

1. **Start CARLA server**:
   ```bash
   cd C:\Program Files\CARLA_0.9.15\WindowsNoEditor
   CarlaUE4.exe -windowed -ResX=800 -ResY=600
   ```

2. **Test Python connection**:
   ```python
   python
   >>> import sys
   >>> sys.path.append("PATH_TO_CARLA_EGG")
   >>> import carla
   >>> client = carla.Client('localhost', 2000)
   >>> client.set_timeout(10.0)
   >>> world = client.get_world()
   >>> print("✓ Connected to CARLA")
   ```

### Test 2: Environment Creation

```bash
python
>>> from src.carla_lane_keeping_env import CarlaLaneKeepingEnv
>>> env = CarlaLaneKeepingEnv()
# Should print: "Connecting to CARLA server..."
#              "✓ CARLA connected successfully"
>>> env.close()
```

### Test 3: Quick Training Run

```bash
python src/training_lane_keeping_ppo.py
# Enter "5000" when asked for timesteps
# Let it run for ~2 minutes
# Press Ctrl+C to stop
```

If all tests pass: **✓ Setup complete!**

---

## Common Issues

### Issue 1: "ImportError: No module named 'carla'"

**Cause**: CARLA egg path not configured correctly

**Fix**:
1. Find the egg file location
2. Update `CARLA_EGG_PATH` in parameters file
3. Use absolute path with escaped backslashes on Windows

### Issue 2: "Connection refused" when connecting to CARLA

**Cause**: CARLA server not running

**Fix**:
1. Start CARLA: `CarlaUE4.exe`
2. Wait 10-20 seconds for server to initialize
3. Retry training

### Issue 3: "CUDA out of memory"

**Cause**: GPU VRAM insufficient for batch size

**Fix**:
```python
# In lane_keeping_parameters.py
BATCH_SIZE = 256  # Reduce from 512
IMAGE_HEIGHT = 96  # Reduce from 128
IMAGE_WIDTH = 96
```

### Issue 4: Python version mismatch

**Error**: `carla-0.9.15-py3.7-*.egg` but you have Python 3.8+

**Fix**: 
- **Must use Python 3.7** for CARLA 0.9.15
- Create new venv with 3.7
- Cannot use newer Python versions

### Issue 5: Training very slow

**Causes**:
1. Using CPU instead of GPU
2. CARLA running in high-res mode
3. Too many background processes

**Fixes**:
1. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
2. Launch CARLA in low-res: `CarlaUE4.exe -ResX=640 -ResY=480`
3. Close other applications

### Issue 6: "Vehicle spawn collision"

**Cause**: Spawn point blocked by static obstacle

**Fix**:
```python
# In lane_keeping_parameters.py
SPAWN_POINT_INDEX = None  # Use random spawn (default)
# Or try different index:
SPAWN_POINT_INDEX = 42  # Try different numbers
```

### Issue 7: Training crashes after X steps

**Cause**: RAM exhaustion or CARLA instability

**Fix**:
1. Reduce `FRAME_STACK` from 4 to 3
2. Restart CARLA server every 100k steps
3. Monitor RAM with Task Manager

---

## Multi-Agent Specific Setup

If using multi-agent version:

### Additional Configuration

```python
# In lane_keeping_parameters.py
class MultiAgentParams:
    ENABLE_MULTI_AGENT = True
    NUM_AGENTS = 2  # Start with 2, increase to 3 if RAM allows
```

### RAM Requirements Check

Before training with N agents:

| Agents | Required RAM | Check Command |
|--------|-------------|---------------|
| 1 | 8GB | - |
| 2 | 12GB | Monitor during first episode |
| 3 | 16GB | Stop if usage >90% |

**Monitor RAM**: 
- Windows: Task Manager → Performance → Memory
- Linux: `htop`

---

## Next Steps

After successful setup:

1. **Train initial model**: Run for 50k steps to verify everything works
2. **Monitor training**: Open TensorBoard to view metrics
3. **Tune parameters**: Adjust reward weights based on observed behavior
4. **Scale up**: Increase to full 200k timesteps training  (or more, based on the outcomes)

---

## Support

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/YOUR_USERNAME/carla-lane-keeping-ppo/issues)
2. Review CARLA documentation
3. Open a new issue with:
   - Error message
   - System specs
   - Steps to reproduce

---

**Setup Time Estimate**: 30-60 minutes for first-time setup

Good luck with your training!
