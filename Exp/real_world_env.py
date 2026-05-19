"""
Real-world environment for cross-scale modular magnetic robot control.



Conventions / failure behavior:
- Observation is an 8D vector with fixed semantic order (see `realEnv` docstring below).
- Action is a 4D normalized vector in [0, 1]. It is converted to physical units before
  commanding the coil: flux(mT), freq(Hz), pitch(deg), direction(deg).
- Missing detections are encoded as coordinates [-1, -1]. Some tasks may encode "robot not present"
  by setting target_location_* to -1; callers may propagate this convention into observations/logs.
"""

import numpy as np
import gym
from gym.spaces import Box
from scipy.interpolate import make_interp_spline
from utils import *
from devices import *
import torch
import cv2
import threading

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

model = None
model_yolo = None

device = 'cuda' if torch.cuda.is_available else 'cpu'
print('\n\nDevice Used:', device)

video_writer_origin = None
video_writer_frame = None
video_writer_frame_no_text = None
video_writer_without = None
video_frame_size = None
video_output_dir = None
video_current_output_dir = None
video_last_episode_number = None
video_last_task_info = None


def calculate_curve_length(x_smooth, y_smooth):
    """Arc length of a parametric curve."""
    length = 0.0
    for i in range(1, len(x_smooth)):
        segment_length = np.sqrt((x_smooth[i] - x_smooth[i - 1])**2 + (y_smooth[i] - y_smooth[i - 1])**2)
        length += segment_length
    return length

def median_filter(data, window_size):
    """Median filter with boundary preservation."""
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size // 2 or i >= len(data) - window_size // 2:
            smoothed_data.append(data[i])
        else:
            smoothed_data.append(np.median(data[i - window_size // 2:i + window_size // 2 + 1], axis=0))
    return smoothed_data

def moving_average(data, window_size):
    """Moving average smoothing."""
    if len(data) < window_size:
        return data
    return [np.mean(data[i:i + window_size], axis=0) for i in range(len(data) - window_size + 1)]

def find_nearest_point(millicore_coord, x_smooth, y_smooth):
    """
    Return the nearest point on the curve, its tangent angle (radians),
    and the normalized arc-length coordinate in [0, 1].
    """
    distances = np.sqrt((x_smooth - millicore_coord[0]) ** 2 + (y_smooth - millicore_coord[1]) ** 2)
    nearest_index = np.argmin(distances)
    nearest_point = (x_smooth[nearest_index], y_smooth[nearest_index])

    if nearest_index > 0 and nearest_index < len(x_smooth) - 1:
        dx = (x_smooth[nearest_index + 1] - x_smooth[nearest_index - 1]) / 2
        dy = (y_smooth[nearest_index + 1] - y_smooth[nearest_index - 1]) / 2
    elif nearest_index == 0:
        dx = x_smooth[1] - x_smooth[0]
        dy = y_smooth[1] - y_smooth[0]
    else:
        dx = x_smooth[-1] - x_smooth[-2]
        dy = y_smooth[-1] - y_smooth[-2]

    tangent_angle = np.arctan2(dy, dx)

    lengths = np.sqrt(np.diff(x_smooth) ** 2 + np.diff(y_smooth) ** 2)
    total_length = np.sum(lengths)

    arc_length = np.sum(lengths[:nearest_index]) + np.sqrt(
        (x_smooth[nearest_index] - x_smooth[nearest_index - 1]) ** 2 + (
                y_smooth[nearest_index] - y_smooth[nearest_index - 1]) ** 2)
    linear_coordinate = arc_length / total_length

    return nearest_point, tangent_angle, linear_coordinate

class realEnv(gym.Env):
    """
    Real-world environment for dual-robot magnetic control.
    
    Observation (8D):
    - [0] T_milli: target location for the milli robot (normalized arc-length in [0, 1], or -1 if not applicable)
    - [1] T_nano: target location for the nano robot (normalized arc-length in [0, 1], or -1 if not applicable)
    - [2] T_release: target release amount/rate (project-specific units; derived from grayscale via `gray_scale2release_rate`)
    - [3] C_milli: current location of the milli robot (normalized arc-length in [0, 1], or -1 if not detected)
    - [4] C_nano: current location of the nano robot (normalized arc-length in [0, 1], or -1 if not detected)
    - [5] C_release: current release amount/rate (same units as T_release)
    - [6] theta_milli: local tangent angle (radians, [0, 2π])
    - [7] theta_nano: local tangent angle (radians, [0, 2π])

    Action (4D, normalized in [0, 1]):
    - [0] flux_density
    - [1] frequency
    - [2] pitch_angle
    - [3] direction_angle
    """
    def __init__(self):
        self.observation_space = self.create_space(
            lows=[-1, -1, -1, -1, -1, -1, -1, -1],
            highs=[1, 1, 1, 1, 1, 1, 2 * np.pi, 2 * np.pi],
            dims=8
        )
        self.action_space = self.create_space(
            lows=[0, 0, 0, 0],
            highs=[1, 1, 1, 1],
            dims=4
        )

    def create_space(self, lows, highs, dims):
        low = np.array(lows, dtype=np.float32)
        high = np.array(highs, dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def activate_devices(self):
        """
        Initialize camera and capture initial environment information.

        Side effects:
        - opens/configures the camera
        - computes pipe centerlines/splines and masks used by `reset()`/`step()`

        Failure behavior:
        - if camera initialization or feature extraction fails, subsequent calls may error or
          return placeholder values; this module does not currently implement a full fallback path.
        """
        self.camera = Camera()
        self.camera.camera_setting()
        # These parameters are part of the experimental vision protocol (geometry + detection threshold).
        self.origin_frame, self.init_frame, self.milli_coords, self.nano_coords, self.mean_gray, self.x_smooth, self.y_smooth, self.spline, self.x_smooth_2, self.y_smooth_2, self.spline_2, self.pipe_mask, self.milli_mask, self.nano_mask, self.intersection_mask = self.camera.get_init_info(pipe_width=0.035, circle_diameter=0.1, threshold_nano=100)


    def reset(self, episode_number, target_location_milli=0, target_location_nano=0, target_release_amount=0, plot_nano=True):
        """
        Reset the environment for a new segment and return the initial observation.

        Args:
            episode_number: segment/episode identifier used for video/log bookkeeping.
            target_location_milli: desired milli position along the pipe centerline, normalized to [0, 1].
            target_location_nano: desired nano position along the pipe centerline, normalized to [0, 1].
            target_release_amount: desired release level (same units as `gray_scale2release_rate` output).
            plot_nano: whether to render/overlay nano-robot info in visualization.

        Returns:
            np.ndarray of shape (8,) following the observation convention documented in `realEnv`.

        Side effects:
            - (re)initializes and stops the coil briefly to enter a known-safe state
            - queries the camera for the latest frame and detections
        """
        self.milli_trajectory = []
        self.nano_trajectory = []
        self.choose_path_milli = 0
        self.choose_path_nano = 0
        self.initial_time = 0

        self.episode_number = episode_number
        # Coil is re-initialized and explicitly stopped to avoid stale state carrying across segments.
        self.coil = Coil()
        cv2.waitKey(200)
        self.coil.stop()
        cv2.waitKey(200)

        self.coil = Coil()

        self.plot_nano = plot_nano
        self.step_number = 0
        self.target_location_milli = target_location_milli
        self.target_location_nano = target_location_nano
        self.target_release_amount = target_release_amount
        [self.current_location_milli, self.current_location_nano, self.current_release_amount, self.total_release_amount, self.theta_milli, self.theta_nano] = np.zeros(6)

        self.origin_frame, self.frame, self.milli_coords, self.nano_coords, self.mean_gray, self.milli_mask, self.nano_mask, self.intersection_mask = self.camera.get_info(choose_path_milli=self.choose_path_milli, choose_path_nano=self.choose_path_nano)

        self.current_release_amount = gray_scale2release_rate(self.mean_gray)

        # Convention: [-1, -1] encodes "no detection" from the vision pipeline.
        unique_milli_coords = np.array([-1, -1])
        unique_nano_coords = np.array([-1, -1])

        if self.milli_coords.size > 0:
            if self.milli_coords.ndim == 2:
                unique_milli_coords = self.milli_coords[0]
            elif self.milli_coords.ndim == 1:
                unique_milli_coords = self.milli_coords

        if self.nano_coords.size > 0:
            if self.nano_coords.ndim == 2:
                unique_nano_coords = self.nano_coords[0]
            elif self.nano_coords.ndim == 1:
                unique_nano_coords = self.nano_coords
        if self.choose_path_milli == 0:
            milli_nearest_point, milli_tangent_angle, self.current_location_milli = find_nearest_point(unique_milli_coords, self.x_smooth, self.y_smooth)
        else:
            milli_nearest_point, milli_tangent_angle, self.current_location_milli = find_nearest_point(unique_milli_coords, self.x_smooth_2, self.y_smooth_2)
        if self.choose_path_nano == 0:
            nano_nearest_point, nano_tangent_angle, self.current_location_nano = find_nearest_point(unique_nano_coords, self.x_smooth, self.y_smooth)
        else:
            nano_nearest_point, nano_tangent_angle, self.current_location_nano = find_nearest_point(unique_nano_coords, self.x_smooth_2, self.y_smooth_2)

        (self.millicore_x, self.millicore_y) = milli_nearest_point
        (self.nanounits_x, self.nanounits_y) = nano_nearest_point

        self.calculate_theta()

        action = [0.0, 0.0, 0.0, 0.0]

        result_image = render_environment(False, 0, 0, 0, self.origin_frame, self.init_frame, action, self.target_location_milli, self.target_location_nano,
                           self.target_release_amount, self.current_location_milli, self.millicore_x, self.millicore_y, self.milli_trajectory, self.current_location_nano, self.nanounits_x, self.nanounits_y, self.nano_trajectory,
                           self.current_release_amount, self.theta_milli, self.theta_nano, self.pipe_mask,
                           self.milli_mask, self.nano_mask, self.intersection_mask, self.x_smooth, self.y_smooth, self.x_smooth_2, self.y_smooth_2,
                           self.choose_path_milli, self.choose_path_nano, self.plot_nano)

        self.current_obs = np.array([self.target_location_milli, self.target_location_nano, self.target_release_amount,
                                     self.current_location_milli, self.current_location_nano,
                                     self.current_release_amount, self.theta_milli, self.theta_nano])

        self.start_time = time.time()
        self.last_time = 0
        self.last_distance_location_milli = 0
        self.last_distance_location_nano = 0
        self.last_distance_release = 0

        self.unique_milli_coords = np.array([-1, -1])
        self.unique_nano_coords = np.array([-1, -1])

        return self.current_obs

    def update_condition(self, target_location_milli, target_location_nano, target_release_amount, plot_nano, choose_path_milli, choose_path_nano):
        """Update target and control-path configuration for this segment."""
        self.plot_nano = plot_nano
        self.target_location_milli = target_location_milli
        self.target_location_nano = target_location_nano
        self.target_release_amount = target_release_amount
        self.choose_path_milli = choose_path_milli
        self.choose_path_nano = choose_path_nano

    def fresh_time(self):
        """Reset the segment start time."""
        self.initial_time = time.time()

    def start_new_video_segment(self, segment_index, task_info):
        """Start a new video segment (and finalize the previous one if any)."""
        if segment_index > 0:
            finalize_video()
        self.video_segment_index = segment_index
        self.video_task_info = task_info
        self.video_new_segment_flag = True

    def step(self, action):
        """
        Apply one action to the coil, resense the environment, and compute reward/dones.

        Args:
            action: normalized action in [0, 1]^4 in the order
                [flux_density, frequency, pitch_angle, direction_angle].

        Returns:
            observation: np.ndarray, shape (8,)
            reward: float
            done: bool (reserved; currently unused/always False in this implementation)
            done_A: bool, task-completion flag used by the higher-level segment loop

        Side effects:
            - commands the physical coil
            - queries the camera
            - renders and may write video frames via `render_environment()`
        """
        # Normalized -> physical units: flux(mT), freq(Hz), pitch(deg), direction(deg).
        coil_action = action * np.array([20, 40, 180, 360])

        self.coil.update_field(new_flux=coil_action[0], new_freq=coil_action[1], new_pitch=coil_action[2], new_direct=coil_action[3])

        self.origin_frame, self.frame, self.milli_coords, self.nano_coords, self.mean_gray, self.milli_mask, self.nano_mask, self.intersection_mask = self.camera.get_info(choose_path_milli=self.choose_path_milli, choose_path_nano=self.choose_path_nano)

        self.current_release_amount = gray_scale2release_rate(self.mean_gray)

        if self.milli_coords.size > 0:
            if self.milli_coords.ndim == 2 and len(self.milli_coords) > 0:
                first_coord = self.milli_coords[0]
                if isinstance(first_coord, np.ndarray):
                    if not np.all(first_coord == -1):
                        self.unique_milli_coords = first_coord
                else:
                    self.unique_milli_coords = first_coord
            elif self.milli_coords.ndim == 1:
                if not np.all(self.milli_coords == -1):
                    self.unique_milli_coords = self.milli_coords

        if self.nano_coords.size > 0:
            if self.nano_coords.ndim == 2:
                self.unique_nano_coords = self.nano_coords[0]
            elif self.nano_coords.ndim == 1:
                self.unique_nano_coords = self.nano_coords
        if self.choose_path_milli == 0:
            milli_nearest_point, milli_tangent_angle, self.current_location_milli = find_nearest_point(self.unique_milli_coords, self.x_smooth, self.y_smooth)
        else:
            milli_nearest_point, milli_tangent_angle, self.current_location_milli = find_nearest_point(self.unique_milli_coords, self.x_smooth_2, self.y_smooth_2)
        if self.choose_path_nano == 0:
            nano_nearest_point, nano_tangent_angle, self.current_location_nano = find_nearest_point(self.unique_nano_coords, self.x_smooth, self.y_smooth)
        else:
            nano_nearest_point, nano_tangent_angle, self.current_location_nano = find_nearest_point(self.unique_nano_coords, self.x_smooth_2, self.y_smooth_2)

        (self.millicore_x, self.millicore_y) = milli_nearest_point
        (self.nanounits_x, self.nanounits_y) = nano_nearest_point

        # Heuristic gating to suppress large detection jumps between frames.
        # This stabilizes trajectories when the vision output occasionally glitches.
        self.std_dev = 0.3

        if len(self.milli_trajectory) == 0:
            self.milli_trajectory.append((self.millicore_x, self.millicore_y))
        else:
            last_point = self.milli_trajectory[-1]
            distance = np.linalg.norm(np.array((self.millicore_x, self.millicore_y)) - np.array(last_point))

            if distance <= self.std_dev:
                self.milli_trajectory.append((self.millicore_x, self.millicore_y))
            else:
                self.millicore_x, self.millicore_y = last_point
                self.milli_trajectory.append(last_point)

        self.milli_trajectory.append((self.millicore_x, self.millicore_y))
        self.nano_trajectory.append((self.nanounits_x, self.nanounits_y))

        self.calculate_theta()

        self.current_obs = np.array([self.target_location_milli, self.target_location_nano, self.target_release_amount, self.current_location_milli, self.current_location_nano, self.current_release_amount, self.theta_milli, self.theta_nano])

        self.current_time = time.time() - self.start_time

        new_episode = getattr(self, 'video_new_segment_flag', False)
        seg_idx = getattr(self, 'video_segment_index', 0)
        out_dir = getattr(self, 'video_output_dir', None)
        task_info = getattr(self, 'video_task_info', None)
        self.result_image = render_environment(new_episode, seg_idx, self.current_time, self.current_time, self.origin_frame, self.frame, action, self.target_location_milli, self.target_location_nano,
                           self.target_release_amount, self.current_location_milli, self.millicore_x, self.millicore_y, self.milli_trajectory, self.current_location_nano, self.nanounits_x, self.nanounits_y, self.nano_trajectory,
                           self.current_release_amount, self.theta_milli, self.theta_nano, self.pipe_mask,
                           self.milli_mask, self.nano_mask, self.intersection_mask, self.x_smooth, self.y_smooth, self.x_smooth_2, self.y_smooth_2,
                           self.choose_path_milli, self.choose_path_nano, self.plot_nano,
                           output_dir=out_dir, task_info=task_info)
        if new_episode:
            self.video_new_segment_flag = False

        distance_location_milli = np.abs(self.current_location_milli - self.target_location_milli)
        distance_location_nano = np.abs(self.current_location_nano - self.target_location_nano)
        distance_release = np.abs(self.current_release_amount - self.target_release_amount)

        reward_distance = -(distance_location_milli ** 2 + distance_location_nano ** 2 + distance_release) ** 0.5

        self.step_time = self.current_time - self.last_time

        self.total_time = time.time() - self.initial_time

        reward_velocity = 30*((self.last_distance_location_milli - distance_location_milli) + (self.last_distance_location_nano - distance_location_nano) + (self.last_distance_release - distance_release))

        reward = 10 * (-self.step_time + reward_distance + reward_velocity)

        self.last_time = self.current_time
        self.last_distance_location_milli = distance_location_milli
        self.last_distance_location_nano = distance_location_nano
        self.last_distance_release = distance_release

        done = False
        done_A = False

        if (distance_location_nano <= 0.05) and (distance_location_milli <= 0.1):
            print('Both robots reached the target!')
            done_A = True
        if self.target_location_milli == -1 and distance_location_nano <= 0.05:
            print('Nano robot reached the target!')
            done_A = True
        if self.target_location_nano == -1 and (distance_location_milli <= 0.1):
            print('Milli robot reached the target!')
            done_A = True
        

        self.step_number += 1

        if done:
            cv2.waitKey(200)
            self.coil.stop()
            cv2.waitKey(200)

        return self.current_obs, reward, done, done_A

    def render(self):
        pass

    def calculate_theta(self):
        """Compute tangent angles for both robots via numerical differentiation."""
        delta = 1e-4

        if self.choose_path_milli == 0:
            millicore_y_plus = self.spline(self.millicore_x + delta)
            millicore_y_minus = self.spline(self.millicore_x - delta)
        else:
            millicore_y_plus = self.spline_2(self.millicore_x + delta)
            millicore_y_minus = self.spline_2(self.millicore_x - delta)

        if self.choose_path_nano == 0:
            nanounits_y_plus = self.spline(self.nanounits_x + delta)
            nanounits_y_minus = self.spline(self.nanounits_x - delta)
        else:
            nanounits_y_plus = self.spline_2(self.nanounits_x + delta)
            nanounits_y_minus = self.spline_2(self.nanounits_x - delta)

        dx_milli = 2 * delta
        dy_milli = millicore_y_plus - millicore_y_minus
        dx_nano = 2 * delta
        dy_nano = nanounits_y_plus - nanounits_y_minus

        if dy_milli >= 0:
            if self.current_location_milli <= self.target_location_milli:
                self.theta_milli = np.arctan2(dy_milli, dx_milli)
            else:
                self.theta_milli = np.arctan2(dy_milli, dx_milli) + np.pi
        else:
            if self.current_location_milli <= self.target_location_milli:
                self.theta_milli = 2 * np.pi + np.arctan2(dy_milli, dx_milli)
            else:
                self.theta_milli = np.pi + np.arctan2(dy_milli, dx_milli)

        if dy_nano >= 0:
            if self.current_location_nano <= self.target_location_nano:
                self.theta_nano = np.arctan2(dy_nano, dx_nano)
            else:
                self.theta_nano = np.arctan2(dy_nano, dx_nano) + np.pi
        else:
            if self.current_location_nano <= self.target_location_nano:
                self.theta_nano = 2 * np.pi + np.arctan2(dy_nano, dx_nano)
            else:
                self.theta_nano = np.pi + np.arctan2(dy_nano, dx_nano)

def render_environment(new_episode, episode_number, frame_number, step_time, origin_frame, frame, action, target_location_milli, target_location_nano, T_release, C_milli, millicore_x, millicore_y, milli_trajectory, C_nano, nanounits_x, nanounits_y, nano_trajectory, C_release, theta_milli, theta_nano, pipe_mask, milli_mask, nano_mask, intersection_mask, x_smooth, y_smooth, x_smooth_2, y_smooth_2, choose_path_milli, choose_path_nano, plot_nano=True, output_dir=None, task_info=None):
    """
    Generate an annotated visualization frame for tracking and control.
    """
    width = frame.shape[1]
    height = frame.shape[0]

    x_smooth_milli = x_smooth
    y_smooth_milli = y_smooth
    x_smooth_nano = x_smooth
    y_smooth_nano = y_smooth

    if choose_path_milli == 1:
        x_smooth_milli = x_smooth_2
        y_smooth_milli = y_smooth_2

    if choose_path_nano == 1:
        x_smooth_nano = x_smooth_2
        y_smooth_nano = y_smooth_2

    curve_length_milli = calculate_curve_length(x_smooth_milli, y_smooth_milli)
    curve_length_nano = calculate_curve_length(x_smooth_nano, y_smooth_nano)

    target_milli_x = target_location_milli * curve_length_milli
    target_nano_x = target_location_nano * curve_length_nano
    cumulative_length = 0.0
    for i in range(1, len(x_smooth_milli)):
        segment_length = np.sqrt((x_smooth_milli[i] - x_smooth_milli[i - 1]) ** 2 + (y_smooth_milli[i] - y_smooth_milli[i - 1]) ** 2)
        if cumulative_length + segment_length >= target_milli_x:
            t = (target_milli_x - cumulative_length) / segment_length
            target_milli_x = x_smooth_milli[i - 1] + t * (x_smooth_milli[i] - x_smooth_milli[i - 1])
            target_milli_y = y_smooth_milli[i - 1] + t * (y_smooth_milli[i] - y_smooth_milli[i - 1])
            break
        cumulative_length += segment_length

    cumulative_length = 0.0
    for i in range(1, len(x_smooth_nano)):
        segment_length = np.sqrt((x_smooth_nano[i] - x_smooth_nano[i - 1]) ** 2 + (y_smooth_nano[i] - y_smooth_nano[i - 1]) ** 2)
        if cumulative_length + segment_length >= target_nano_x:
            t = (target_nano_x - cumulative_length) / segment_length
            target_nano_x = x_smooth_nano[i - 1] + t * (x_smooth_nano[i] - x_smooth_nano[i - 1])
            target_nano_y = y_smooth_nano[i - 1] + t * (y_smooth_nano[i] - y_smooth_nano[i - 1])
            break
        cumulative_length += segment_length
    target_milli_x_pixel = int(target_milli_x * width)
    target_milli_y_pixel = int(target_milli_y * height)
    target_nano_x_pixel = int(target_nano_x * width)
    target_nano_y_pixel = int(target_nano_y * height)

    overlay = frame.copy()
    overlay_without_pipe = frame.copy()
    pipe_mask_not = cv2.bitwise_not(pipe_mask)
    overlay[pipe_mask_not == 255] = (0, 0, 0)
    overlay[milli_mask == 255] = (176, 129, 30)
    overlay[nano_mask == 255] = (81, 81, 236)
    overlay[intersection_mask == 255] = (255, 255, 0)

    overlay_without_pipe[milli_mask == 255] = (176, 129, 30)
    overlay_without_pipe[nano_mask == 255] = (81, 81, 236)
    overlay_without_pipe[intersection_mask == 255] = (255, 255, 0)

    if (np.abs(target_location_milli - C_milli) <= 0.15):
        circle_mask_target_milli = np.zeros((height, width), dtype=np.uint8)
        if (np.abs(target_location_milli - C_milli) - 0.06) > 0:
            cv2.circle(circle_mask_target_milli, (target_milli_x_pixel, target_milli_y_pixel), int((np.abs(target_location_milli - C_milli)-0.06) * width), 255, -1)
            target_milli_mask = cv2.bitwise_and(circle_mask_target_milli, pipe_mask)
            b_channel = 255*(10*(np.abs(target_location_milli - C_milli) - 0.06))
            g_channel = 255*(1-10*(np.abs(target_location_milli - C_milli) - 0.06))
            r_channel = 0
            color_target_milli = [b_channel, g_channel, r_channel]
            overlay[target_milli_mask == 255] = color_target_milli
            overlay_without_pipe[target_milli_mask == 255] = color_target_milli

    if plot_nano:
        if (np.abs(target_location_nano - C_nano) <= 0.15):
            circle_mask_target_nano = np.zeros((height, width), dtype=np.uint8)
            if (np.abs(target_location_nano - C_nano)-0.06)>0:
                cv2.circle(circle_mask_target_nano, (target_nano_x_pixel, target_nano_y_pixel), int((np.abs(target_location_nano - C_nano)-0.06) * width), 255, -1)  # filled circle
                target_nano_mask = cv2.bitwise_and(circle_mask_target_nano, pipe_mask)
                b_channel = 0
                g_channel = 255*(1-10*(np.abs(target_location_milli - C_milli) - 0.06))
                r_channel = 255*(10*(np.abs(target_location_milli - C_milli) - 0.06))
                color_target_nano = [b_channel, g_channel, r_channel]
                overlay[target_nano_mask == 255] = color_target_nano
                overlay_without_pipe[target_nano_mask == 255] = color_target_nano
    result_image = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
    result_without_pipe = cv2.addWeighted(frame, 0.5, overlay_without_pipe, 0.5, 0)
    if (np.abs(target_location_milli - C_milli) - 0.06) > 0:
        cv2.circle(result_image, (target_milli_x_pixel, target_milli_y_pixel), 5, (176, 129, 30), -1)
        cv2.circle(result_without_pipe, (target_milli_x_pixel, target_milli_y_pixel), 5, (176, 129, 30), -1)
    else:
        cv2.circle(result_image, (target_milli_x_pixel, target_milli_y_pixel), 5, (0, 255, 0), -1)
        cv2.circle(result_without_pipe, (target_milli_x_pixel, target_milli_y_pixel), 5, (0, 255, 0), -1)

    # Keep a text-free copy of frame-version video before any putText overlays.
    result_image_no_text = result_image.copy()

    cv2.putText(result_image, f'Target milli: {float(target_location_milli):.2f}',
                (width - 400, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result_without_pipe, f'Target milli: {float(target_location_milli):.2f}',
                (width - 400, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(result_image, f'Target release: {float(T_release):.2f}',
                (width - 400, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result_without_pipe, f'Target release: {float(T_release):.2f}',
                (width - 400, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(result_image, f'Current milli: {float(C_milli):.2f}',
                (width - 200, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result_without_pipe, f'Current milli: {float(C_milli):.2f}',
                (width - 200, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(result_image, f'Current release: {float(C_release):.2f}',
                (width - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result_without_pipe, f'Current release: {float(C_release):.2f}',
                (width - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    millicore_x_pixel = int(millicore_x * width)
    millicore_y_pixel = int((millicore_y) * height)

    arrow_length = 50
    arrow_x_milli = int(millicore_x_pixel + arrow_length * np.cos(theta_milli))
    arrow_y_milli = int(millicore_y_pixel + arrow_length * np.sin(theta_milli))
    cv2.arrowedLine(result_image, (millicore_x_pixel, millicore_y_pixel), (arrow_x_milli, arrow_y_milli), (176, 129, 30), 2)
    cv2.arrowedLine(result_image_no_text, (millicore_x_pixel, millicore_y_pixel), (arrow_x_milli, arrow_y_milli), (176, 129, 30), 2)
    cv2.arrowedLine(result_without_pipe, (millicore_x_pixel, millicore_y_pixel), (arrow_x_milli, arrow_y_milli), (176, 129, 30), 2)
    if plot_nano:
        if (np.abs(target_location_nano - C_nano) - 0.06) > 0:
            cv2.circle(result_image, (target_nano_x_pixel, target_nano_y_pixel), 5, (81, 81, 236), -1)
            cv2.circle(result_image_no_text, (target_nano_x_pixel, target_nano_y_pixel), 5, (81, 81, 236), -1)
            cv2.circle(result_without_pipe, (target_nano_x_pixel, target_nano_y_pixel), 5, (81, 81, 236), -1)
        else:
            cv2.circle(result_image, (target_nano_x_pixel, target_nano_y_pixel), 5, (0, 255, 0), -1)
            cv2.circle(result_image_no_text, (target_nano_x_pixel, target_nano_y_pixel), 5, (0, 255, 0), -1)
            cv2.circle(result_without_pipe, (target_nano_x_pixel, target_nano_y_pixel), 5, (0, 255, 0), -1)

        cv2.putText(result_image, f'Target nano: {float(target_location_nano):.2f}',
                    (width - 400, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(result_without_pipe, f'Target nano: {float(target_location_nano):.2f}',
                    (width - 400, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(result_image, f'Current nano: {float(C_nano):.2f}',
                    (width - 200, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(result_without_pipe, f'Current nano: {float(C_nano):.2f}',
                    (width - 200, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        nanounits_x_pixel = int(nanounits_x * width)
        nanounits_y_pixel = int((nanounits_y) * height)

        arrow_x_nano = int(nanounits_x_pixel + arrow_length * np.cos(theta_nano))
        arrow_y_nano = int(nanounits_y_pixel + arrow_length * np.sin(theta_nano))

        cv2.arrowedLine(result_image, (nanounits_x_pixel, nanounits_y_pixel), (arrow_x_nano, arrow_y_nano), (81, 81, 236), 2)
        cv2.arrowedLine(result_image_no_text, (nanounits_x_pixel, nanounits_y_pixel), (arrow_x_nano, arrow_y_nano), (81, 81, 236), 2)
        cv2.arrowedLine(result_without_pipe, (nanounits_x_pixel, nanounits_y_pixel), (arrow_x_nano, arrow_y_nano), (81, 81, 236), 2)

    window_size = 10
    if len(milli_trajectory) > window_size:
        smoothed_milli_trajectory = moving_average(milli_trajectory, window_size)
    else:
        smoothed_milli_trajectory = milli_trajectory

    milli_num_points_to_show = min(50, len(smoothed_milli_trajectory))

    for i in range(milli_num_points_to_show):
        milli_point = smoothed_milli_trajectory[-(milli_num_points_to_show - i)]
        t = i / milli_num_points_to_show
        b_value = int(176 + (255 - 176) * t)
        g_value = int(129 + (200 - 129) * t)
        r_value = int(30 + (100 - 30) * t)
        milli_color = (b_value, g_value, r_value)

        cv2.circle(result_image, (int(milli_point[0] * width), int(milli_point[1] * height)), 3, milli_color, -1)
        cv2.circle(result_image_no_text, (int(milli_point[0] * width), int(milli_point[1] * height)), 3, milli_color, -1)
        cv2.circle(result_without_pipe, (int(milli_point[0] * width), int(milli_point[1] * height)), 3, milli_color, -1)
    if plot_nano:
        if len(nano_trajectory) > window_size:
            smoothed_nano_trajectory = moving_average(nano_trajectory, window_size)
        else:
            smoothed_nano_trajectory = nano_trajectory
        nano_num_points_to_show = min(50, len(smoothed_nano_trajectory))

        for i in range(nano_num_points_to_show):
            nano_point = smoothed_nano_trajectory[-(nano_num_points_to_show - i)]
            t = i / nano_num_points_to_show
            r_value = int(81 + (236 - 81) * t)
            g_value = int(81 + (81 - 81) * t)
            b_value = int(236 + (81 - 236) * t)
            nano_color = (b_value, g_value, r_value)

            cv2.circle(result_image, (int(nano_point[0] * width), int(nano_point[1] * height)), 3, nano_color, -1)
            cv2.circle(result_image_no_text, (int(nano_point[0] * width), int(nano_point[1] * height)), 3, nano_color, -1)
            cv2.circle(result_without_pipe, (int(nano_point[0] * width), int(nano_point[1] * height)), 3, nano_color, -1)
    if new_episode:
        if episode_number != 0:
            finalize_video()
        if origin_frame is not None and len(origin_frame.shape) >= 2:
            height, width = origin_frame.shape[:2]
            frame_size = (width, height)
        else:
            frame_size = (500, 500)
        initialize_video_writers(episode_number, frame_size, output_dir=output_dir, task_info=task_info)

    save_frame_to_video(origin_frame, episode_number, video_writer_origin)
    save_frame_to_video(result_image, episode_number, video_writer_frame)
    save_frame_to_video(result_image_no_text, episode_number, video_writer_frame_no_text)
    save_frame_to_video(result_without_pipe, episode_number, video_writer_without)

    return result_image


def calculate_action(action, theta_milli, theta_nano):
    """
    Decompose global action into local robot frames.
    
    Transforms the global direction angle to local coordinates based on each robot's
    tangent angle while preserving flux, frequency, and pitch parameters.
    """
    if (theta_milli >= 0) and (theta_milli < np.pi/2):
        angle_milli = theta_milli
    elif (theta_milli >= np.pi/2) and (theta_milli < np.pi):
        angle_milli = theta_milli - np.pi
    elif (theta_milli >= np.pi) and (theta_milli < 3/2*np.pi):
        angle_milli = theta_milli - np.pi
    elif (theta_milli >= 3/2*np.pi) and (theta_milli < 2*np.pi):
        angle_milli = theta_milli - 2*np.pi
    else:
        angle_milli = 0

    if (theta_nano >= 0) and (theta_nano < np.pi / 2):
        angle_nano = theta_nano
    elif (theta_nano >= np.pi / 2) and (theta_nano < np.pi):
        angle_nano = theta_nano - np.pi
    elif (theta_nano >= np.pi) and (theta_nano < 3 / 2 * np.pi):
        angle_nano = theta_nano - np.pi
    elif (theta_nano >= 3 / 2 * np.pi) and (theta_nano < 2 * np.pi):
        angle_nano = theta_nano - 2*np.pi
    else:
        angle_nano = 0

    action_milli = np.asarray(action, dtype=np.float64).copy()
    action_nano = np.asarray(action, dtype=np.float64).copy()

    global_direction = action_milli[3] * 2 * np.pi
    milli_direction = ((global_direction - angle_milli) + (8 * np.pi)) % (2 * np.pi)
    nano_direction = ((global_direction - angle_nano) + (8 * np.pi)) % (2 * np.pi)

    action_milli[3] = milli_direction / (2*np.pi)
    action_nano[3] = nano_direction / (2*np.pi)

    return action_milli, action_nano

def initialize_video_writers(episode_number, frame_size, fps=10, output_dir=None, task_info=None):
    """Initialize video writers and metadata tracking."""
    global video_writer_origin, video_writer_frame, video_writer_frame_no_text, video_writer_without, video_frame_size
    global video_last_episode_number, video_last_task_info, video_current_output_dir
    if not (isinstance(frame_size, tuple) and len(frame_size) == 2):
        if hasattr(frame_size, 'shape'):
            height, width = frame_size.shape[:2]
            frame_size = (width, height)
        else:
            print("Warning: failed to determine frame size, using default (500, 500)")
            frame_size = (500, 500)

    out_dir = output_dir if output_dir is not None else video_output_dir
    if out_dir is None:
        out_dir = os.path.join('tmp', 'video')
    os.makedirs(out_dir, exist_ok=True)
    video_last_episode_number = episode_number
    video_last_task_info = task_info
    video_current_output_dir = out_dir

    video_filename_origin = f"episode_{episode_number}_origin.avi"
    video_file_path_origin = os.path.join(out_dir, video_filename_origin)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer_origin = cv2.VideoWriter(video_file_path_origin, fourcc, fps, frame_size)
    
    if not video_writer_origin.isOpened():
        print(f"Error: failed to create video file {video_file_path_origin}")
        video_file_path_origin = video_file_path_origin.replace('.avi', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_origin = cv2.VideoWriter(video_file_path_origin, fourcc, fps, frame_size)
        if not video_writer_origin.isOpened():
            print(f"Error: failed to create MP4 video file {video_file_path_origin}")
    video_filename_frame = f"episode_{episode_number}_frame.avi"
    video_file_path_frame = os.path.join(out_dir, video_filename_frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer_frame = cv2.VideoWriter(video_file_path_frame, fourcc, fps, frame_size)

    if not video_writer_frame.isOpened():
        print(f"Error: failed to create video file {video_file_path_frame}")
        video_file_path_frame = video_file_path_frame.replace('.avi', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_frame = cv2.VideoWriter(video_file_path_frame, fourcc, fps, frame_size)
        if not video_writer_frame.isOpened():
            print(f"Error: failed to create MP4 video file {video_file_path_frame}")

    video_filename_frame_no_text = f"episode_{episode_number}_frame_no_text.avi"
    video_file_path_frame_no_text = os.path.join(out_dir, video_filename_frame_no_text)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer_frame_no_text = cv2.VideoWriter(video_file_path_frame_no_text, fourcc, fps, frame_size)

    if not video_writer_frame_no_text.isOpened():
        print(f"Error: failed to create video file {video_file_path_frame_no_text}")
        video_file_path_frame_no_text = video_file_path_frame_no_text.replace('.avi', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_frame_no_text = cv2.VideoWriter(video_file_path_frame_no_text, fourcc, fps, frame_size)
        if not video_writer_frame_no_text.isOpened():
            print(f"Error: failed to create MP4 video file {video_file_path_frame_no_text}")

    video_filename_without = f"episode_{episode_number}_without.avi"
    video_file_path_without = os.path.join(out_dir, video_filename_without)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer_without = cv2.VideoWriter(video_file_path_without, fourcc, fps, frame_size)
    
    if not video_writer_without.isOpened():
        print(f"Error: failed to create video file {video_file_path_without}")
        video_file_path_without = video_file_path_without.replace('.avi', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_without = cv2.VideoWriter(video_file_path_without, fourcc, fps, frame_size)
        if not video_writer_without.isOpened():
            print(f"Error: failed to create MP4 video file {video_file_path_without}")
    
    video_frame_size = frame_size
    print(f"Video writers initialized, frame size: {frame_size}, FPS: {fps}")

def save_frame_to_video(image, episode_number, video_writer):
    """Write a single frame to a video writer."""
    global video_writer_origin, video_writer_frame, video_writer_frame_no_text, video_writer_without, video_frame_size

    if video_writer is None:
        if image is not None and len(image.shape) >= 2:
            height, width = image.shape[:2]
            frame_size = (width, height)
        else:
            frame_size = (500, 500)
        initialize_video_writers(episode_number, frame_size)
    if video_writer is None or not video_writer.isOpened():
        print("Warning: video writer not initialized or invalid, skipping this frame")
        return
    if image is not None and video_frame_size is not None:
        actual_height, actual_width = image.shape[:2]
        expected_width, expected_height = video_frame_size
        
        if actual_width != expected_width or actual_height != expected_height:
            image = cv2.resize(image, (expected_width, expected_height))
        
        try:
            success = video_writer.write(image)
            if not success:
                print("Warning: failed to write video frame")
        except Exception as e:
            print(f"Warning: exception occurred while writing video frame: {e}")
    else:
        if image is None:
            print("Warning: image is None, skipping write")
        elif video_frame_size is None:
            print("Warning: video frame size is not set, skipping write")
def _write_video_task_info_txt():
    """Write segment task metadata to a text file."""
    global video_last_episode_number, video_last_task_info, video_current_output_dir
    if video_current_output_dir is None or video_last_task_info is None:
        return
    import datetime
    base = f"episode_{video_last_episode_number}"
    txt_path = os.path.join(video_current_output_dir, f"{base}_info.txt")
    lines = [
        f"# Task info (corresponds to {base}_origin.avi / {base}_frame.avi / {base}_frame_no_text.avi / {base}_without.avi)",
        f"# Generated at: {datetime.datetime.now().isoformat()}",
        "",
    ]
    for k, v in video_last_task_info.items():
        if isinstance(v, (list, tuple)):
            v = str(v)
        lines.append(f"{k}: {v}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Task info written to: {txt_path}")


def finalize_video():
    """Release video writers and persist task metadata."""
    global video_writer_origin, video_writer_frame, video_writer_frame_no_text, video_writer_without
    _write_video_task_info_txt()
    if video_writer_origin is not None:
        video_writer_origin.release()
        video_writer_origin = None
    if video_writer_frame is not None:
        video_writer_frame.release()
        video_writer_frame = None
    if video_writer_frame_no_text is not None:
        video_writer_frame_no_text.release()
        video_writer_frame_no_text = None
    if video_writer_without is not None:
        video_writer_without.release()
        video_writer_without = None