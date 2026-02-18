"""
CARLA Lane Keeping PPO Training Script
Train PPO agent for autonomous lane keeping in CARLA
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Forces non-interactive backend (prevents Tcl error)
# ---------------------------------------------------

import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings("ignore", message=".*sys.meta_path is None.*")

# Minimal, clean style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 100

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

# Torch imports
import torch
import torch.nn as nn
import gymnasium as gym

# Local imports
from carla_lane_keeping_env import CarlaLaneKeepingEnv
from lane_keeping_parameters import *

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for processing stacked camera frames
    Combined with vehicle state information
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Extract dimensions
        n_input_frames = observation_space['image'].shape[0]  # Frame stack size
        image_height = observation_space['image'].shape[1]
        image_width = observation_space['image'].shape[2]
        n_channels = observation_space['image'].shape[3]
        
        vehicle_state_dim = observation_space['vehicle_state'].shape[0]
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv3d(n_channels, 32, kernel_size=(1, 8, 8), stride=(1, 4, 4), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            
            # Second conv block
            nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            # Third conv block
            nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            # Adaptive pooling to ensure consistent output size
            nn.AdaptiveAvgPool3d((1, 4, 4))  # Output: (batch, 128, 1, 4, 4)
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.zeros(1, n_channels, n_input_frames, image_height, image_width)
            sample_input = sample_input.permute(0, 1, 2, 3, 4)  # Rearrange for 3D conv
            cnn_output_size = self.cnn(sample_input).view(1, -1).shape[1]
        
        # Fully connected layers
        self.fc_image = nn.Sequential(
            nn.Linear(cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Vehicle state processing
        self.fc_state = nn.Sequential(
            nn.Linear(vehicle_state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined feature processing
        combined_dim = 256 + 64  # CNN features + state features
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, observations):
        # Extract image and vehicle state
        images = observations['image'].float() / 255.0  # Normalize to [0, 1]
        vehicle_state = observations['vehicle_state'].float()
        
        # Rearrange image dimensions for 3D CNN: (batch, channels, frames, height, width)
        images = images.permute(0, 4, 1, 2, 3)
        
        # Process images through CNN
        cnn_features = self.cnn(images)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
        image_features = self.fc_image(cnn_features)
        
        # Process vehicle state
        state_features = self.fc_state(vehicle_state)
        
        # Combine features
        combined_features = torch.cat([image_features, state_features], dim=1)
        final_features = self.fc_combined(combined_features)
        
        return final_features

class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    """Custom policy using our CNN feature extractor"""
    
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = CustomCNNFeatureExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 512}
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring and logging"""
    
    def __init__(self, save_path, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.lane_deviations = []
        self.collision_rates = []
        
        # Create plots directory
        self.plots_dir = os.path.join(save_path, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Training metrics
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get info from the environment
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            # Log episode completion
            if 'episode' in info.keys():
                self.episode_count += 1
                episode_reward = info.get('episode_reward', 0)
                episode_length = info.get('step', 0)
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Calculate moving averages
                window_size = min(100, len(self.episode_rewards))
                mean_reward = np.mean(self.episode_rewards[-window_size:])
                mean_length = np.mean(self.episode_lengths[-window_size:])
                
                if self.verbose > 0 and self.episode_count % TrainingParams.LOG_INTERVAL == 0:
                    print(f"Episode {self.episode_count:4d} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Mean Reward: {mean_reward:7.2f} | "
                          f"Episode Length: {episode_length:4d} | "
                          f"Mean Length: {mean_length:6.1f}")
                
                # Update best reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"New best mean reward: {self.best_mean_reward:.2f}")
                
                # Generate plots periodically
                if self.episode_count % DebugParams.PLOT_UPDATE_FREQ == 0:
                    self._generate_training_plots()
        
        return True
    
    def _generate_training_plots(self):
        """Generate training progress plots"""
        if len(self.episode_rewards) < 10:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f'Training Progress — Episode {self.episode_count}', fontsize=14)

        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        episodes = np.arange(len(rewards))

        def smooth(data, window=20):
            if len(data) >= window:
                return np.convolve(data, np.ones(window)/window, mode='valid'), np.arange(window-1, len(data))
            return None, None

        # --- Plot 1: Episode Reward ---
        ax = axes[0, 0]
        ax.plot(episodes, rewards, alpha=0.3, color='royalblue', linewidth=0.8)
        ma, ma_x = smooth(rewards)
        if ma is not None:
            ax.plot(ma_x, ma, color='royalblue', linewidth=2, label='MA(20)')
        ax.set_title('Episode Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Episode Length ---
        ax = axes[0, 1]
        ax.plot(episodes, lengths, alpha=0.3, color='darkorange', linewidth=0.8)
        ma, ma_x = smooth(lengths.astype(float))
        if ma is not None:
            ax.plot(ma_x, ma, color='darkorange', linewidth=2, label='MA(20)')
        ax.set_title('Episode Length')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Plot 3: Rolling Mean Reward (multiple windows) ---
        ax = axes[1, 0]
        for window, color, lbl in [(10, 'red', 'MA(10)'), (50, 'green', 'MA(50)'), (100, 'blue', 'MA(100)')]:
            ma, ma_x = smooth(rewards, window)
            if ma is not None:
                ax.plot(ma_x, ma, color=color, linewidth=1.8, label=lbl)
        ax.set_title('Rolling Mean Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Plot 4: Success Rate (episodes reaching ≥90% of max length) ---
        ax = axes[1, 1]
        if len(lengths) >= 50:
            max_len = lengths.max()
            threshold = max_len * 0.9
            window = 50
            sr_x, sr_y = [], []
            for i in range(window, len(lengths)):
                w = lengths[i-window:i]
                sr_x.append(i)
                sr_y.append(100 * np.mean(w >= threshold))
            ax.plot(sr_x, sr_y, color='seagreen', linewidth=2)
            ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_ylim(0, 100)
            ax.set_title(f'Success Rate (steps ≥ {threshold:.0f})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Rate (%)')
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'training_progress_episode_{self.episode_count}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save training data
        training_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_mean_reward': self.best_mean_reward,
            'episode_count': self.episode_count
        }
        data_path = os.path.join(self.save_path, 'training_data.json')
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)


def generate_final_training_plots(callback_data, save_path):
    """Generate final training summary plots in a clean, TensorBoard-style layout"""

    if len(callback_data.episode_rewards) < 10:
        print("Not enough data for final plots")
        return

    rewards = np.array(callback_data.episode_rewards)
    lengths = np.array(callback_data.episode_lengths)
    episodes = np.arange(len(rewards))

    def smooth(data, window):
        if len(data) >= window:
            return np.convolve(data, np.ones(window)/window, mode='valid'), np.arange(window-1, len(data))
        return None, None

    n = len(rewards)
    split = n // 3

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('PPO Lane Keeping — Final Training Analysis', fontsize=15, fontweight='bold')

    # --- 1. Episode Reward ---
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.25, color='royalblue', linewidth=0.7)
    for window, color, lbl in [(20, 'royalblue', 'MA(20)'), (100, 'navy', 'MA(100)')]:
        ma, ma_x = smooth(rewards, window)
        if ma is not None:
            ax.plot(ma_x, ma, color=color, linewidth=2, label=lbl)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('Episode Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 2. Episode Length ---
    ax = axes[0, 1]
    ax.plot(episodes, lengths, alpha=0.25, color='darkorange', linewidth=0.7)
    for window, color, lbl in [(20, 'darkorange', 'MA(20)'), (100, 'saddlebrown', 'MA(100)')]:
        ma, ma_x = smooth(lengths.astype(float), window)
        if ma is not None:
            ax.plot(ma_x, ma, color=color, linewidth=2, label=lbl)
    ax.axhline(MAX_EPISODE_STEPS, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Max Steps')
    ax.set_title('Episode Length')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Rolling Mean Reward (multiple windows — TensorBoard style) ---
    ax = axes[1, 0]
    for window, color, lbl in [(10, '#e74c3c', 'MA(10)'), (50, '#2ecc71', 'MA(50)'), (100, '#3498db', 'MA(100)')]:
        ma, ma_x = smooth(rewards, window)
        if ma is not None:
            ax.plot(ma_x, ma, color=color, linewidth=1.8, label=lbl)
    ax.set_title('Rolling Mean Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. Success Rate over time ---
    ax = axes[1, 1]
    max_len = lengths.max()
    threshold = max_len * 0.9
    window = 50
    sr_x, sr_y = [], []
    for i in range(window, n):
        w = lengths[i-window:i]
        sr_x.append(i)
        sr_y.append(100 * np.mean(w >= threshold))
    if sr_y:
        ax.plot(sr_x, sr_y, color='seagreen', linewidth=2)
        ax.fill_between(sr_x, sr_y, alpha=0.15, color='seagreen')
        ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='50% baseline')
    ax.set_ylim(0, 100)
    ax.set_title(f'Success Rate (steps ≥ {threshold:.0f}, window=50)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 5. Reward Distribution: Early vs Mid vs Late (overlaid histograms) ---
    ax = axes[2, 0]
    early  = rewards[:split]
    mid    = rewards[split:2*split]
    late   = rewards[2*split:]
    bins = np.linspace(rewards.min(), rewards.max(), 40)
    ax.hist(early, bins=bins, alpha=0.5, color='#e74c3c', label=f'Early (ep 0–{split})')
    ax.hist(mid,   bins=bins, alpha=0.5, color='#f39c12', label=f'Mid   (ep {split}–{2*split})')
    ax.hist(late,  bins=bins, alpha=0.5, color='#2ecc71', label=f'Late  (ep {2*split}–{n})')
    ax.axvline(np.mean(early), color='#c0392b', linestyle='--', linewidth=1.2)
    ax.axvline(np.mean(late),  color='#27ae60', linestyle='--', linewidth=1.2)
    ax.set_title('Reward Distribution: Early vs Mid vs Late')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 6. Training Statistics Text Summary ---
    ax = axes[2, 1]
    ax.axis('off')
    stats = (
        f"{'TRAINING SUMMARY':^40}\n"
        f"{'─'*40}\n"
        f"  Total Episodes       : {n}\n"
        f"  Total Timesteps      : {n * int(np.mean(lengths)):,}\n\n"
        f"  Reward  — Mean       : {np.mean(rewards):.2f}\n"
        f"  Reward  — Std        : {np.std(rewards):.2f}\n"
        f"  Reward  — Best       : {rewards.max():.2f}\n"
        f"  Reward  — Worst      : {rewards.min():.2f}\n\n"
        f"  Final 100 ep avg     : {np.mean(rewards[-100:]):.2f}\n"
        f"  Best mean reward     : {callback_data.best_mean_reward:.2f}\n\n"
        f"  Ep Length — Mean     : {np.mean(lengths):.1f}\n"
        f"  Ep Length — Max      : {lengths.max()}\n"
        f"  Success rate (last 100): {100*np.mean(lengths[-100:] >= threshold):.1f}%\n"
    )
    ax.text(0.05, 0.97, stats, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#aaa', alpha=0.9))

    plt.tight_layout()
    final_plot_path = os.path.join(save_path, 'final_training_analysis.png')
    plt.savefig(final_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Individual plots for reward and length
    for data, color, dark_color, title, ylabel, fname in [
        (rewards, 'royalblue', 'navy', 'Episode Reward', 'Total Reward', 'final_episode_rewards.png'),
        (lengths.astype(float), 'darkorange', 'saddlebrown', 'Episode Length', 'Steps', 'final_episode_lengths.png'),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data, alpha=0.25, color=color, linewidth=0.7)
        ma, ma_x = smooth(data, 20)
        if ma is not None:
            ax.plot(ma_x, ma, color=color, linewidth=2, label='MA(20)')
        ma, ma_x = smooth(data, 100)
        if ma is not None:
            ax.plot(ma_x, ma, color=dark_color, linewidth=2, label='MA(100)')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, fname), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"✓ Final training plots saved to: {save_path}")
    print(f"  - final_training_analysis.png")
    print(f"  - final_episode_rewards.png, final_episode_lengths.png")

def create_env(rank=0):
    """Create and wrap the environment"""
    def _init():
        env = CarlaLaneKeepingEnv()
        env = Monitor(env)
        return env
    set_random_seed(rank)
    return _init

def load_existing_model():
    """Ask user if they want to load an existing model"""
    model_path = input("Enter path to existing model (or press Enter for new training): ").strip()
    
    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            model = PPO.load(model_path)
            print("✓ Model loaded successfully!")
            return model, model_path
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Starting new training instead...")
            return None, None
    elif model_path:
        print(f"✗ Model file not found: {model_path}")
        print("Starting new training instead...")
        return None, None
    else:
        print("Starting new training...")
        return None, None

def setup_training_environment():
    """Setup directories and training environment"""
    # Validate parameters
    validate_parameters()
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(MODELS_PATH, f"ppo_lane_keeping_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Training run directory: {run_dir}")
    
    # Save parameters for this run
    params_file = os.path.join(run_dir, "training_parameters.json")
    params_dict = {
        "PPO": {
            "learning_rate": PPOParams.LEARNING_RATE,
            "clip_range": PPOParams.CLIP_RANGE,
            "entropy_coef": PPOParams.ENTROPY_COEF,
            "value_function_coef": PPOParams.VALUE_FUNCTION_COEF,
            "max_grad_norm": PPOParams.MAX_GRAD_NORM,
            "n_steps": PPOParams.N_STEPS,
            "batch_size": PPOParams.BATCH_SIZE,
            "n_epochs": PPOParams.N_EPOCHS,
            "gamma": PPOParams.GAMMA,
            "gae_lambda": PPOParams.GAE_LAMBDA
        },
        "Training": {
            "total_timesteps": TrainingParams.TOTAL_TIMESTEPS,
            "save_freq": TrainingParams.SAVE_FREQ,
            "eval_freq": TrainingParams.EVAL_FREQ
        },
        "Environment": {
            "max_episode_steps": MAX_EPISODE_STEPS,
            "target_speed": TARGET_SPEED,
            "min_speed_threshold": MIN_SPEED_THRESHOLD
        }
    }
    
    # Setup tensorboard logging
    tensorboard_log = os.path.join(run_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log, exist_ok=True)

    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    return run_dir

def create_ppo_model(env, existing_model=None, tensorboard_log=None):
    """Create PPO model with custom policy"""
    
    if existing_model is not None:
        # Update environment for existing model
        existing_model.set_env(env)
        return existing_model
    
    # Create new model
    policy_kwargs = {
        'net_arch': PPOParams.NET_ARCH,
        'activation_fn': getattr(nn, PPOParams.ACTIVATION_FN.upper()) if hasattr(nn, PPOParams.ACTIVATION_FN.upper()) else nn.Tanh,
        'features_extractor_class': CustomCNNFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 512}
    }
    
    model = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=PPOParams.LEARNING_RATE,
        tensorboard_log=tensorboard_log,
        n_steps=PPOParams.N_STEPS,
        batch_size=PPOParams.BATCH_SIZE,
        n_epochs=PPOParams.N_EPOCHS,
        gamma=PPOParams.GAMMA,
        gae_lambda=PPOParams.GAE_LAMBDA,
        clip_range=PPOParams.CLIP_RANGE,
        ent_coef=PPOParams.ENTROPY_COEF,
        vf_coef=PPOParams.VALUE_FUNCTION_COEF,
        max_grad_norm=PPOParams.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=TrainingParams.VERBOSE,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return model

def main():
    """Main training function"""
    print("="*50)
    print("CARLA Lane Keeping PPO Training")
    print("="*50)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup training environment
    run_dir = setup_training_environment()
    
    # Ask about loading existing model
    existing_model, model_path = load_existing_model()
    
    # Create environment
    print("Creating CARLA environment...")
    try:
        # Single environment for now (can be extended to multiple environments)
        env = DummyVecEnv([create_env(i) for i in range(1)])
        print("✓ Environment created successfully")
        
        # Create evaluation environment
        eval_env = DummyVecEnv([create_env(1000)])  # Different seed for eval
        
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        print("Make sure CARLA server is running!")
        return
    
    # Create PPO model
    print("Creating PPO model...")
    model = create_ppo_model(env, existing_model, os.path.join(run_dir, "tensorboard_logs"))
    print("✓ Model created successfully")
    print(f"Model device: {model.device}")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingParams.SAVE_FREQ,
        save_path=run_dir,
        name_prefix='ppo_lane_keeping'
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=TrainingParams.EVAL_FREQ,
        n_eval_episodes=TrainingParams.N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Custom training callback
    training_callback = TrainingCallback(run_dir, verbose=1)
    callbacks.append(training_callback)
    
    # Start training
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    print(f"Total timesteps: {TrainingParams.TOTAL_TIMESTEPS:,}")
    print(f"Save frequency: {TrainingParams.SAVE_FREQ:,} timesteps")
    print(f"Evaluation frequency: {TrainingParams.EVAL_FREQ:,} timesteps")
    print(f"Models will be saved to: {run_dir}")
    print("\nTraining Controls:")
    print("- Press Ctrl+C to pause training and save current progress")
    print("- You can resume later by loading the saved model")
    print("="*50)

    start_time = time.time()
    current_timesteps = 0
    training_segments = []

    while current_timesteps < TrainingParams.TOTAL_TIMESTEPS:
        remaining_steps = TrainingParams.TOTAL_TIMESTEPS - current_timesteps
        
        # Ask user how many steps to train this segment
        try:
            segment_steps = input(f"\nCurrent progress: {current_timesteps:,}/{TrainingParams.TOTAL_TIMESTEPS:,} timesteps\n"
                                 f"Remaining: {remaining_steps:,} timesteps\n"
                                 f"Enter timesteps for this training segment (or 'all' for remaining): ").strip()
            
            if segment_steps.lower() == 'all':
                steps_to_train = remaining_steps
            else:
                steps_to_train = min(int(segment_steps), remaining_steps)
        
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or interrupted. Using 50,000 steps as default.")
            steps_to_train = min(50000, remaining_steps)
        
        print(f"\nTraining {steps_to_train:,} timesteps...")
        segment_start = time.time()
        
        try:
            model.learn(
                total_timesteps=steps_to_train,
                callback=callbacks,
                log_interval=TrainingParams.LOG_INTERVAL,
                progress_bar=True,
                reset_num_timesteps=False  # Continue counting from previous segments
            )
            
            current_timesteps += steps_to_train
            segment_time = time.time() - segment_start
            training_segments.append({
                'steps': steps_to_train,
                'time': segment_time,
                'total_steps': current_timesteps
            })
            
            # Save checkpoint after each segment
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{current_timesteps}_steps")
            model.save(checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            print(f"Segment completed in {segment_time/60:.1f} minutes")
            
            if current_timesteps >= TrainingParams.TOTAL_TIMESTEPS:
                break
                
            # Ask if user wants to continue
            continue_training = input("\nContinue training? (y/n/tune): ").strip().lower()
            if continue_training == 'n':
                print("Training stopped by user.")
                break
            elif continue_training == 'tune':
                print("You can now tune parameters in lane_keeping_parameters.py")
                print("Restart the script and load the checkpoint to continue with new parameters.")
                break
                
        except KeyboardInterrupt:
            print(f"\n⚠ Training interrupted by user at {current_timesteps:,} timesteps")
            interrupted_path = os.path.join(run_dir, f"interrupted_{current_timesteps}_steps")
            model.save(interrupted_path)
            print(f"Model saved: {interrupted_path}")
            break
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            error_path = os.path.join(run_dir, f"error_{current_timesteps}_steps")
            model.save(error_path)
            print(f"Model saved: {error_path}")
            raise

    try:
        # Final model save
        if current_timesteps >= TrainingParams.TOTAL_TIMESTEPS:
            final_model_path = os.path.join(run_dir, "final_model")
            model.save(final_model_path)
            total_time = time.time() - start_time
            print(f"\n✓ Training completed in {total_time/3600:.2f} hours")
            print(f"Final model saved to: {final_model_path}")

        # Print training summary
        print(f"\nTraining Summary:")
        print(f"Total timesteps trained: {current_timesteps:,}")
        print(f"Training segments: {len(training_segments)}")
        for i, segment in enumerate(training_segments):
            print(f"  Segment {i+1}: {segment['steps']:,} steps in {segment['time']/60:.1f} min")

        # Generate final plots
        print("\nGenerating final training plots...")
        generate_final_training_plots(training_callback, run_dir)
    
    finally:
            # Cleanup
            try:
                # Disable tqdm before cleanup to prevent shutdown errors
                import tqdm
                tqdm.tqdm._instances.clear()
                
                env.close()
                eval_env.close()
                print("✓ Environment cleanup completed")
            except Exception as e:
                print(f"Cleanup warning: {e}")
                pass


if __name__ == "__main__":
    main()