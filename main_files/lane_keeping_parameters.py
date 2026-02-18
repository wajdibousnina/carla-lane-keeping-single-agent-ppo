"""
Lane Keeping Parameters Configuration
For CARLA PPO Lane Keeping Project
"""

import os

# ==================== PATHS ====================
BASE_PATH = "F:\\thesis_carla_lane_keeping"
MODELS_PATH = os.path.join(BASE_PATH, "models")
CARLA_EGG_PATH = "F:\\Program files\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg"

# Ensure directories exist
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# ==================== CARLA CONNECTION ====================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0 # was 10.0
CARLA_MAP = "Town04"  # Town01, Town02, Town03, Town04, Town05, Town10HD, etc.
# ==================== ENVIRONMENT PARAMETERS ====================
# Episode settings
MAX_EPISODE_STEPS = 2000
MIN_SPEED_THRESHOLD = 1.0  # m/s - prevents reward hacking (was 0.1)
TARGET_SPEED = 8.0  # m/s - approximately 30 km/h

# Spawn settings
SPAWN_POINT_INDEX = None  # None for random, or specific index
WEATHER_RANDOMIZATION = False

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 90
SPECTATOR_HEIGHT = 10.0
SPECTATOR_DISTANCE = 8.0

# ==================== REWARD FUNCTION PARAMETERS ====================
class RewardParams:
    # Lane keeping rewards
    LANE_CENTER_WEIGHT = -1.0          # Penalty for distance from lane center
    FORWARD_PROGRESS_WEIGHT = 20.0      # Reward for forward movement (was 5.0)
    ORIENTATION_WEIGHT = -0.5          # Penalty for yaw error
    SPEED_REWARD_WEIGHT = 5.0          # Reward for maintaining good speed (was 1.0)
    
    # Penalty for being stopped
    MIN_SPEED_PENALTY = -5.0           # Penalty for being stopped (small value, was -0.1)

    # Safety penalties
    COLLISION_PENALTY = -50.0         # Large penalty for collision (was -100.0)
    LANE_DEPARTURE_PENALTY = -20.0     # Penalty for leaving lane completely (was -50.0)
    REVERSE_PENALTY = -5.0            # Penalty for going backwards (was -10.0)
    
    # Action smoothness
    STEERING_SMOOTHNESS_WEIGHT = -0.1  # Penalty for abrupt steering changes
    THROTTLE_SMOOTHNESS_WEIGHT = -0.05 # Penalty for abrupt throttle changes
    
    # Completion bonus
    COMPLETION_BONUS = 60.0           # Bonus for completing episode successfully

# ==================== OBSERVATION SPACE ====================
class ObservationParams:
    # Image processing
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    FRAME_STACK = 4  # Number of frames to stack for temporal information
    
    # Normalization parameters
    SPEED_NORMALIZATION = 15.0  # Max expected speed for normalization
    ANGLE_NORMALIZATION = 1.0   # Already in radians
    DISTANCE_NORMALIZATION = 4.0  # Max lane width for normalization

# ==================== ACTION SPACE ====================
class ActionParams:
    # Continuous action space: [steering, throttle, brake]
    STEERING_RANGE = (-1.0, 1.0)      # Full steering range
    THROTTLE_RANGE = (0.0, 1.0)       # Throttle range
    BRAKE_RANGE = (0.0, 1.0)          # Brake range
    
    # Action smoothing parameters
    STEERING_SMOOTHING = 0.3           # Reduced from 0.1
    THROTTLE_SMOOTHING = 0.3           # Reduced from 0.1
    BRAKE_SMOOTHING = 0.3              # Reduced from 0.1
    
    # Maximum action changes per step (prevents jerky movements)
    MAX_STEERING_CHANGE = 0.5          # Increased from 0.3
    MAX_THROTTLE_CHANGE = 0.8          # Increased from 0.5
    MAX_BRAKE_CHANGE = 0.8             # Increased from 0.5

# ==================== PPO HYPERPARAMETERS ====================
class PPOParams:
    # Learning parameters
    LEARNING_RATE = 5e-4            # It was 3e-4
    LR_SCHEDULE = 'linear'  # 'linear', 'constant', or 'cosine'
    
    # PPO specific parameters
    CLIP_RANGE = 0.2                   # PPO clip parameter
    ENTROPY_COEF = 0.02                # Initial entropy coefficient (it was 0.05)
    ENTROPY_COEF_FINAL = 0.005         # Final entropy coefficient (it was 0.001)
    VALUE_FUNCTION_COEF = 0.5          # Value function loss coefficient
    MAX_GRAD_NORM = 0.5                # Gradient clipping
    
    # Training parameters
    N_STEPS = 4096                     # Steps per update (Increased from 2048)
    BATCH_SIZE = 512                   # Mini-batch size (Increased from 64 to 256 or 512)
    N_EPOCHS = 4                       # Training epochs per update
    GAE_LAMBDA = 0.95                  # GAE lambda parameter
    GAMMA = 0.99                       # Discount factor
    
    # Network architecture
    NET_ARCH = [256, 256]              # Hidden layer sizes
    ACTIVATION_FN = 'tanh'             # 'tanh', 'relu', etc.

# ==================== TRAINING PARAMETERS ====================
class TrainingParams:
    TOTAL_TIMESTEPS = 200_000        # Total training timesteps (reduced from 1M)
    SAVE_FREQ = 10_000                 # Save model every N timesteps (was 50k)
    EVAL_FREQ = 5_000                 # Evaluate every N timesteps (was 25k)
    N_EVAL_EPISODES = 5                # Number of evaluation episodes
    
    # Early stopping
    PATIENCE = 20_000                 # Timesteps to wait for improvement (was 100k)
    MIN_IMPROVEMENT = 0.01             # Minimum improvement threshold
    
    # Logging
    LOG_INTERVAL = 10                  # Log every N updates
    TENSORBOARD_LOG = True             # Enable tensorboard logging
    VERBOSE = 1                        # Verbosity level

# ==================== CURRICULUM LEARNING ====================
class CurriculumParams:
    ENABLE_CURRICULUM = True
    
    # Stage 1: Straight roads, clear weather
    STAGE_1_EPISODES = 200
    STAGE_1_WEATHER_VARIANCE = 0.0
    STAGE_1_TRAFFIC_DENSITY = 0.0
    
    # Stage 2: Slight curves, some weather variation
    STAGE_2_EPISODES = 300
    STAGE_2_WEATHER_VARIANCE = 0.3
    STAGE_2_TRAFFIC_DENSITY = 0.1
    
    # Stage 3: Complex roads, full variation
    STAGE_3_EPISODES = float('inf')  # Continue until training ends
    STAGE_3_WEATHER_VARIANCE = 1.0
    STAGE_3_TRAFFIC_DENSITY = 0.3

# ==================== DEBUGGING AND MONITORING ====================
class DebugParams:
    SAVE_EPISODE_REPLAYS = False       # Save episode recordings
    PRINT_EPISODE_STATS = True         # Print episode statistics
    SAVE_EPISODE_PLOTS = True          # Save training plots
    PLOT_UPDATE_FREQ = 100             # Update plots every N episodes
    
    # Metrics to track
    TRACK_LANE_DEVIATION = True
    TRACK_SPEED_PROFILE = True
    TRACK_ACTION_SMOOTHNESS = True
    TRACK_REWARD_COMPONENTS = True

# ==================== VALIDATION ====================
def validate_parameters():
    """Validate parameter consistency"""
    errors = []
    
    # Check paths exist
    if not os.path.exists(os.path.dirname(CARLA_EGG_PATH)):
        errors.append(f"CARLA egg path directory does not exist: {os.path.dirname(CARLA_EGG_PATH)}")
    
    # Check parameter ranges
    if not (0 < PPOParams.CLIP_RANGE < 1):
        errors.append("PPO clip range must be between 0 and 1")
    
    if PPOParams.ENTROPY_COEF < PPOParams.ENTROPY_COEF_FINAL:
        errors.append("Initial entropy coefficient must be >= final entropy coefficient")
    
    if PPOParams.BATCH_SIZE > PPOParams.N_STEPS:
        errors.append("Batch size cannot be larger than n_steps")
    
    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(errors))
    
    print("✓ All parameters validated successfully")

if __name__ == "__main__":
    validate_parameters()
    print("Lane keeping parameters loaded successfully!")