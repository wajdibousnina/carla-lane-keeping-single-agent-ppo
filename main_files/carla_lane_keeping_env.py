"""
CARLA Lane Keeping Environment
Gymnasium-compatible environment for PPO training
"""

import sys
import os
import random
import math
import numpy as np
import cv2
from collections import deque
import time

# Add CARLA to path
sys.path.append("F:\\Program files\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg")
import carla

import gymnasium as gym
from gymnasium import spaces

from lane_keeping_parameters import *

class CarlaLaneKeepingEnv(gym.Env):
    """CARLA Environment for Lane Keeping using PPO"""
    
    def __init__(self):
        super(CarlaLaneKeepingEnv, self).__init__()
        
        # Environment parameters
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.current_step = 0
        self.episode_count = 0
        self.stuck_time = 0
        
        # CARLA connection
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.map = None
        
        # Vehicle and sensors
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        # State tracking
        self.camera_data = None
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.previous_location = None
        self.previous_action = np.array([0.0, 0.0, 0.0])  # [steering, throttle, brake]
        self.frame_buffer = deque(maxlen=ObservationParams.FRAME_STACK)
        
        # Reward tracking
        self.episode_reward = 0.0
        self.distance_from_center_history = []
        
        # Curriculum learning
        self.curriculum_stage = 1
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([ActionParams.STEERING_RANGE[0], ActionParams.THROTTLE_RANGE[0], ActionParams.BRAKE_RANGE[0]]),
            high=np.array([ActionParams.STEERING_RANGE[1], ActionParams.THROTTLE_RANGE[1], ActionParams.BRAKE_RANGE[1]]),
            dtype=np.float32
        )
        
        # Observation space: stacked frames + vehicle state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(ObservationParams.FRAME_STACK, ObservationParams.IMAGE_HEIGHT, ObservationParams.IMAGE_WIDTH, 3),
                dtype=np.uint8
            ),
            'vehicle_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),  # [speed, steering, yaw_rate, distance_to_center, lane_yaw_error, progress]
                dtype=np.float32
            )
        })
        
        # Initialize CARLA
        self._initialize_carla()
    
    def _initialize_carla(self):
        """Initialize CARLA client and world"""
        try:
            print("Connecting to CARLA server...")
            self.client = carla.Client(CARLA_HOST, CARLA_PORT)
            self.client.set_timeout(CARLA_TIMEOUT)
            
            # Load world
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.map = self.world.get_map()

            # Load specific map if different from current
            from lane_keeping_parameters import CARLA_MAP
            current_map_name = self.world.get_map().name
            if CARLA_MAP not in current_map_name:
                print(f"Loading map: {CARLA_MAP}")
                self.world = self.client.load_world(CARLA_MAP)
                time.sleep(2)  # Wait for map to load
                self.blueprint_library = self.world.get_blueprint_library()
                self.map = self.world.get_map()
                print(f"Map loaded: {self.world.get_map().name}")

            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            print("✓ CARLA connected successfully")
            
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            raise
    
    def _setup_vehicle(self):
            """Spawn and setup vehicle with robust retry logic"""
            # 1. Get all spawn points
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available")
            
            # 2. FILTER: Highways (Straight roads are better for learning)
            highway_spawns = []
            for sp in spawn_points:
                # Town04 Highway coordinates approx check
                if (sp.location.y > -50 and sp.location.y < 300 and 
                    abs(sp.location.x) < 100):
                    highway_spawns.append(sp)
            
            # Use highway spawns if available, otherwise fallback to all
            possible_spawns = highway_spawns if highway_spawns else spawn_points

            # 3. RETRY LOOP (The Fix for your crash)
            max_retries = 10
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            
            for attempt in range(max_retries):
                try:
                    # Pick a random spot
                    if SPAWN_POINT_INDEX is not None:
                        spawn_point = spawn_points[SPAWN_POINT_INDEX % len(spawn_points)]
                    else:
                        spawn_point = random.choice(possible_spawns)
                    
                    # Try to spawn
                    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    
                    # If we get here, spawn was successful!
                    break 
                    
                except RuntimeError as e:
                    # If it failed (collision), log it and try again immediately
                    if "collision" in str(e).lower():
                        print(f"⚠ Spawn collision at attempt {attempt+1}/{max_retries}. Retrying...")
                        continue
                    else:
                        raise e # Raise other unexpected errors
            
            # Check if we actually spawned something
            if self.vehicle is None:
                raise RuntimeError(f"Failed to spawn vehicle after {max_retries} attempts")

            # 4. Standard Setup (Physics settling)
            self.world.tick()  # Process the spawn
            time.sleep(0.1)    # Give physics time to settle

            # Ensure vehicle is not in handbrake
            initial_control = carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=0.0,
                hand_brake=False, reverse=False, manual_gear_shift=False
            )
            self.vehicle.apply_control(initial_control)
            self.world.tick()

            # Debug info
            # print(f"Vehicle physics enabled: {self.vehicle.is_alive}")
            
            # Setup sensors
            self._setup_camera_sensor()
            self._setup_collision_sensor()
            self._setup_lane_invasion_sensor()
            
            # Set spectator view
            self._set_spectator_view(spawn_point)
            
            # Initialize vehicle state
            self.previous_location = self.vehicle.get_location()
            self.collision_detected = False
            self.lane_invasion_detected = False
            
            # print(f"✓ Vehicle spawned at location: {spawn_point.location}")
    
    def _setup_camera_sensor(self):
        """Setup RGB camera sensor"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(CAMERA_FOV))
        
        # Mount camera on vehicle (front view)
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(self._on_camera_data)
    
    def _setup_collision_sensor(self):
        """Setup collision sensor"""
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)
    
    def _setup_lane_invasion_sensor(self):
        """Setup lane invasion sensor"""
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)
    
    def _set_spectator_view(self, spawn_point):
        """Set spectator camera behind the vehicle"""
        spectator = self.world.get_spectator()
        
        # Calculate position behind and above the vehicle
        vehicle_transform = spawn_point
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        
        # Convert rotation to radians
        yaw_rad = math.radians(vehicle_rotation.yaw)
        
        # Position behind the vehicle
        spectator_location = carla.Location(
            x=vehicle_location.x - SPECTATOR_DISTANCE * math.cos(yaw_rad),
            y=vehicle_location.y - SPECTATOR_DISTANCE * math.sin(yaw_rad),
            z=vehicle_location.z + SPECTATOR_HEIGHT
        )
        
        # Look towards the vehicle
        spectator_rotation = carla.Rotation(
            pitch=-20.0,
            yaw=vehicle_rotation.yaw,
            roll=0.0
        )
        
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)
        spectator.set_transform(spectator_transform)
        
        print("✓ Spectator camera positioned. Use right-click + WASD for manual control")
    
    def _on_camera_data(self, image):
        """Process camera data"""
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Resize and store
        resized = cv2.resize(array, (ObservationParams.IMAGE_WIDTH, ObservationParams.IMAGE_HEIGHT))
        self.camera_data = resized
    
    def _on_collision(self, event):
        """Handle collision events"""
        self.collision_detected = True
        if DebugParams.PRINT_EPISODE_STATS:
            print(f"Collision detected with {event.other_actor.type_id}")
    
    def _on_lane_invasion(self, event):
        """Handle lane invasion events"""
        self.lane_invasion_detected = True
        if DebugParams.PRINT_EPISODE_STATS:
            try:
                # Try different ways to access lane markings (CARLA version compatibility)
                if hasattr(event, 'crossed_markings'):
                    lane_types = [str(marking.type).split('.')[-1] for marking in event.crossed_markings]
                elif hasattr(event, 'crossed_lane_markings'):
                    lane_types = [str(marking.type).split('.')[-1] for marking in event.crossed_lane_markings]
                else:
                    lane_types = ["Unknown"]
                print(f"Lane invasion detected: {lane_types}")
            except Exception as e:
                print(f"Lane invasion detected (details unavailable): {e}")
    
    def _apply_action(self, action):
        """Apply action to vehicle - simplified for debugging"""
        # Use raw actions with minimal processing
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        
        # FORCE movement: if throttle is low and brake is low, use significant throttle (i'm keeping this till the agent sees that moving forward gives good rewards)
        if throttle < 0.5 and brake < 0.3:
            throttle = 0.7  # Strong throttle
            brake = 0.0     # No brake
        
        # If braking, reduce throttle
        if brake > 0.5:
            throttle = 0.0
        # When the agent start learning agent sees that moving forward gives good rewards i'm gonna use one of these strategies:
        #Strategy 1: Gradual Fade-Out (BEST)
        #Replace the forced throttle with a version that fades out over time:

        ## FORCE movement with gradual fade-out
        # min_throttle = max(0.3, 0.7 - (self.episode_count * 0.001))  # Decreases over episodes

        # if throttle < min_throttle and brake < 0.3:
        #     throttle = min_throttle
        #     brake = 0.0

        # if brake > 0.5:
        #     throttle = 0.0
        #This starts at 0.7 forced throttle, then gradually decreases to 0.3 as training progresses.

        #Strategy 2: Conditional Training Wheels
        #Only force throttle if the agent is REALLY timid:
        
        # # Only intervene if agent is being extremely conservative
        # if throttle < 0.3 and brake < 0.3:  # Changed threshold from 0.5 to 0.3
        #     throttle = 0.5  # Less aggressive override (was 0.7)
        #     brake = 0.0


        # DEBUG: Print every 50 steps
        if hasattr(self, 'current_step') and self.current_step % 50 == 0:
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            location = self.vehicle.get_location()
            print(f"Step {self.current_step}: Action=[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
            print(f"Step {self.current_step}: Control - Steer: {steering:.3f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}")
            print(f"Step {self.current_step}: Speed: {speed:.3f} m/s, Location: ({location.x:.1f}, {location.y:.1f})")
            
            # Check if vehicle is stuck
            if hasattr(self, 'last_location'):
                distance_moved = math.sqrt((location.x - self.last_location.x)**2 + (location.y - self.last_location.y)**2)
                print(f"Step {self.current_step}: Distance moved in 50 steps: {distance_moved:.3f} m")
            self.last_location = location
        
        # Apply control
        control = carla.VehicleControl(
            steer=steering,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        
        self.vehicle.apply_control(control)
        self.previous_action = np.array([steering, throttle, brake])
    
    def _get_vehicle_state(self):
        """Get current vehicle state for observation"""
        # Vehicle physics
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        transform = self.vehicle.get_transform()
        
        # Calculate speed
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get lane information
        waypoint = self.map.get_waypoint(transform.location)
        
        # Distance to lane center
        lane_center = waypoint.transform.location
        vehicle_location = transform.location
        distance_to_center = math.sqrt((vehicle_location.x - lane_center.x)**2 + 
                                     (vehicle_location.y - lane_center.y)**2)
        
        # Yaw error relative to lane
        lane_yaw = waypoint.transform.rotation.yaw
        vehicle_yaw = transform.rotation.yaw
        yaw_error = math.radians(vehicle_yaw - lane_yaw)
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # Normalize to [-π, π]
        
        # Forward progress
        if self.previous_location is not None:
            progress = math.sqrt((vehicle_location.x - self.previous_location.x)**2 + 
                               (vehicle_location.y - self.previous_location.y)**2)
            # Check if moving backwards
            forward_vector = transform.get_forward_vector()
            movement_vector = carla.Vector3D(
                vehicle_location.x - self.previous_location.x,
                vehicle_location.y - self.previous_location.y,
                0
            )
            dot_product = forward_vector.x * movement_vector.x + forward_vector.y * movement_vector.y
            if dot_product < 0:
                progress = -progress
        else:
            progress = 0.0
        
        # Normalize values
        normalized_speed = speed / ObservationParams.SPEED_NORMALIZATION
        normalized_distance = distance_to_center / ObservationParams.DISTANCE_NORMALIZATION
        normalized_yaw_error = yaw_error / ObservationParams.ANGLE_NORMALIZATION
        
        state = np.array([
            normalized_speed,
            self.previous_action[0],  # Current steering
            angular_velocity.z,       # Yaw rate
            normalized_distance,
            normalized_yaw_error,
            progress
        ], dtype=np.float32)
        
        return state, distance_to_center, yaw_error, speed, progress
    
    def _calculate_reward(self, distance_to_center, yaw_error, speed, progress, action):
            """Calculate reward based on current state"""
            reward = 0.0

            # --- REWARD COMPONENT 1: STOPPING PENALTY ---
            # If the car is stopped, give a tiny penalty and return immediately.
            if speed < MIN_SPEED_THRESHOLD:
                # Reward is -0.1 (or whatever MIN_SPEED_PENALTY is set to)
                reward += RewardParams.MIN_SPEED_PENALTY

            # --- REWARD COMPONENT 2: BASE REWARD (New Exploration Incentive) ---
            # A small positive reward for every step the car is moving and not crashing.
            # keeping the agent alive and encourages exploration.
            reward += 0.1

            # --- REWARD COMPONENT 3: FORWARD PROGRESS ---
            # progress is distance moved in this step (m/step).
            # Set FORWARD_PROGRESS_WEIGHT to a high positive number (e.g., 5.0).
            reward += RewardParams.FORWARD_PROGRESS_WEIGHT * progress

            # --- REWARD COMPONENT 4: LANE CENTERING (Gentle Penalty) ---
            # Use squared distance to make the penalty gentle near the center
            # but harsh as the car moves farther out.
            centering_penalty = RewardParams.LANE_CENTER_WEIGHT * (distance_to_center ** 2)
            reward += centering_penalty

            # --- REWARD COMPONENT 5: HEADING/YAW ERROR (Stronger Penalty) ---
            # Penalize bad rotation, but keep it linear (abs)
            reward += RewardParams.ORIENTATION_WEIGHT * abs(yaw_error)

            # --- REWARD COMPONENT 6: SPEED TARGET ---
            # Reward for being close to the TARGET_SPEED
            speed_diff = abs(speed - TARGET_SPEED)
            speed_reward = RewardParams.SPEED_REWARD_WEIGHT * (1.0 - (speed_diff / TARGET_SPEED))
            reward += speed_reward

            # --- REWARD COMPONENT 7: ZIG-ZAG PENALTY (Action/Yaw Rate Smoothness) ---
            # Penalty for high angular velocity (spinning or aggressive turns).
            angular_velocity_z = self.vehicle.get_angular_velocity().z
            reward -= 0.5 * abs(angular_velocity_z)
            
            # --- REWARD COMPONENT 8: ACTION SMOOTHNESS (Penalize Jerky Input) ---
            current_action = np.array([action[0], action[1], action[2]])
            action_diff = np.linalg.norm(current_action - self.previous_action)
            
            # Penalyze large steering changes
            reward += RewardParams.STEERING_SMOOTHNESS_WEIGHT * action_diff

            # Store for next step calculation
            self.previous_action = current_action

            return float(reward)
    
    def _get_observation(self):
        """Get current observation"""
        # Wait for camera data if not available
        timeout = 0
        while self.camera_data is None and timeout < 10:
            self.world.tick()
            time.sleep(0.1)
            timeout += 1
        
        if self.camera_data is None:
            # Fallback: black image
            self.camera_data = np.zeros((ObservationParams.IMAGE_HEIGHT, ObservationParams.IMAGE_WIDTH, 3), dtype=np.uint8)
        
        # Add frame to buffer
        self.frame_buffer.append(self.camera_data.copy())
        
        # Ensure buffer is full
        while len(self.frame_buffer) < ObservationParams.FRAME_STACK:
            self.frame_buffer.append(self.camera_data.copy())
        
        # Stack frames
        stacked_frames = np.stack(list(self.frame_buffer), axis=0)
        
        # Get vehicle state
        vehicle_state, _, _, _, _ = self._get_vehicle_state()
        
        observation = {
            'image': stacked_frames,
            'vehicle_state': vehicle_state
        }
        
        return observation
    
    def _cleanup(self):
        """Clean up CARLA actors"""
        actors_to_destroy = []
        
        if self.camera_sensor is not None:
            actors_to_destroy.append(self.camera_sensor)
        if self.collision_sensor is not None:
            actors_to_destroy.append(self.collision_sensor)
        if self.lane_invasion_sensor is not None:
            actors_to_destroy.append(self.lane_invasion_sensor)
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
        
        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except:
                pass
        
        # Clear references
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_data = None
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Clean up previous episode
        self._cleanup()
        
        # Setup new episode
        self._setup_vehicle()
        
        # Reset state
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.previous_action = np.array([0.0, 0.0, 0.0])
        self.frame_buffer.clear()
        self.distance_from_center_history.clear()
        self.stuck_time = 0
        
        # Set weather based on curriculum
        if WEATHER_RANDOMIZATION:
            self._randomize_weather()
        
        # Get initial observation
        observation = self._get_observation()
        info = {"episode": self.episode_count}
        
        self.episode_count += 1
        return observation, info
    
    def _randomize_weather(self):
        """Randomize weather conditions"""
        weather = carla.WeatherParameters(
            cloudiness=random.uniform(0, 100),
            precipitation=random.uniform(0, 50),
            sun_altitude_angle=random.uniform(30, 90),
            wind_intensity=random.uniform(0, 50)
        )
        self.world.set_weather(weather)
    
    def step(self, action):
        """Execute one environment step"""
        # Apply action
        self._apply_action(action)
        
        # Tick world
        self.world.tick()
        # # Check if vehicle is physically stuck
        # if self.current_step % 100 == 0:  # Every 100 steps
        #     velocity = self.vehicle.get_velocity()
        #     speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        #     if speed < 0.1:  # If basically not moving
        #         print(f"WARNING: Vehicle appears stuck at step {self.current_step}")
        #         # Try to unstick by applying strong throttle
        #         emergency_control = carla.VehicleControl(
        #             throttle=1.0,
        #             steer=0.0,
        #             brake=0.0,
        #             hand_brake=False
        #         )
        #         self.vehicle.apply_control(emergency_control)
        #         for _ in range(5):  # Apply for 5 ticks
        #             self.world.tick()
        # Get new state
        vehicle_state, distance_to_center, yaw_error, speed, progress = self._get_vehicle_state()

        # Update stuck timer
        if speed < MIN_SPEED_THRESHOLD:
            self.stuck_time += 1
        else:
            self.stuck_time = 0
        
        # Calculate reward
        reward = self._calculate_reward(distance_to_center, yaw_error, speed, progress, action)
        
        # Update tracking
        self.episode_reward += reward
        self.distance_from_center_history.append(distance_to_center)
        self.current_step += 1
        self.previous_location = self.vehicle.get_location()

        # Check termination conditions
        terminated = False
        truncated = False
        
        # 1. COLLISION CHECK
        if self.collision_detected:
            terminated = True
            if DebugParams.PRINT_EPISODE_STATS:
                print("Episode terminated: Collision")

        # 2. BAD ORIENTATION CHECK (Prevent Spinning)
        # If car is 60+ degrees (approx 1.0 rad) off angle
        elif abs(yaw_error) > 1.0:
            terminated = True
            reward -= 10.0  # Penalty for losing control
            if DebugParams.PRINT_EPISODE_STATS:
                print("Episode terminated: Bad Orientation")

        # 3. LANE DEPARTURE CHECK
        elif distance_to_center > 3.0:
            terminated = True
            if DebugParams.PRINT_EPISODE_STATS:
                print("Episode terminated: Too far from lane")

        # 4. STUCK TIMEOUT CHECK (Prevent Parking)
        if self.stuck_time > 1000:  # Approx 3 seconds stuck (was 60)
            truncated = True
            reward -= 20.0  # penalty for being stuck
            if DebugParams.PRINT_EPISODE_STATS:
                print("Episode truncated: Vehicle stuck")
        
        # Episode truncation (max steps reached)
        if self.current_step >= self.max_episode_steps:
            truncated = True
            reward += RewardParams.COMPLETION_BONUS  # Bonus for completing episode
            if DebugParams.PRINT_EPISODE_STATS:
                print("Episode completed successfully")
        
        # Get observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            "episode_reward": self.episode_reward,
            "distance_to_center": distance_to_center,
            "speed": speed,
            "yaw_error": yaw_error,
            "progress": progress,
            "collision": self.collision_detected,
            "lane_invasion": self.lane_invasion_detected,
            "step": self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Close environment"""
        self._cleanup()
        if self.world is not None:
            # Restore asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        print("✓ CARLA environment closed")