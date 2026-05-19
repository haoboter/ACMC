"""
Gym-style pipe navigation environment: transition dynamics from SL-trained digital twins of X-bot
behaviors, shaped rewards, OpenCV render.

Weights for those twins live under ``SL_model/AL_all/``.
"""
import gym
from gym.spaces import Box
import torch
import random

from utils.util import *
import numpy as np
from scipy.interpolate import make_interp_spline
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Digital-twin networks for step dynamics (1. millicore / 2. nanounit velocity and 3. reconfiguration-release/assembly).
model_velocity_millicore = NN_motion_prediction()
model_velocity_nano = NN_motion_prediction()
model_release_rate = NN_grayscale_prediction()

if os.path.exists('SL_model/AL_all/milli_velocity.pth'):
    model_velocity_millicore.load_state_dict(torch.load('SL_model/AL_all/milli_velocity.pth', map_location='cpu'))
    model_velocity_millicore.eval()
else:
    print('Millicore velocity prediction model was not found')
if os.path.exists('SL_model/AL_all/nano_velocity.pth'):
    model_velocity_nano.load_state_dict(torch.load('SL_model/AL_all/nano_velocity.pth', map_location='cpu'))
    model_velocity_nano.eval()
else:
    print('Nanounits velocity prediction model was not found')

if os.path.exists('SL_model/AL_all/release_grayscale.pth'):
    model_release_rate.load_state_dict(torch.load('SL_model/AL_all/release_grayscale.pth', map_location='cpu'))
    model_release_rate.eval()
else:
    print('Release-rate prediction model was not found')


class simEnv(gym.Env):
    def __init__(self):
        """Define environment state/action bounds and task-specific thresholds."""
        self.target_location_millicore = 0
        self.target_location_nano = 0
        self.target_release_amount = 0
        self.current_location_millicore = 0
        self.current_location_nano = 0
        self.current_release_amount = 0
        
        # Max |target − average_release| for success; looser when target_release_amount==1 (experimental).
        self.release_thresholds = {0: 0.1, 1: 0.3}

        # Observation encoding convention:
        # -1 indicates task/robot absence, otherwise values are normalized.
        self.target_location_millicore_low = -1
        self.target_location_millicore_high = 1
        self.current_location_millicore_low = -1
        self.current_location_millicore_high = 1

        self.target_location_nano_low = -1
        self.target_location_nano_high = 1
        self.current_location_nano_low = -1
        self.current_location_nano_high = 1

        # Release state is defined in [0, 1].
        # When millicore is absent, release task is treated as inactive.
        self.target_release_amount_low = 0
        self.target_release_amount_high = 1
        self.current_release_amount_low = 0
        self.current_release_amount_high = 1

        # Robot tangent direction in radians.
        self.theta_millicore_low = -1
        self.theta_millicore_high = 2 * np.pi
        self.theta_nano_low = -1
        self.theta_nano_high = 2 * np.pi

        # Observation space (8): target_millicore, target_nano, target_release, current_millicore,
        # current_nano, current_release, theta_millicore, theta_nano.
        obs_low = np.array([self.target_location_millicore_low, self.target_location_nano_low, self.target_release_amount_low, self.current_location_millicore_low, self.current_location_nano_low, self.current_release_amount_low, self.theta_millicore_low, self.theta_nano_low], dtype=np.float32)
        obs_high = np.array([self.target_location_millicore_high, self.target_location_nano_high, self.target_release_amount_high, self.current_location_millicore_high, self.current_location_nano_high, self.current_release_amount_high, self.theta_millicore_high, self.theta_nano_high], dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action components are normalized to [0, 1].
        # Physical units: strength [0,20] mT, frequency [0,40] Hz,
        # pitch [0,180] deg, direction [0,360] deg.
        self.strength_low = 0
        self.strength_high = 1
        self.frequency_low = 0
        self.frequency_high = 1
        self.pitch_low = 0
        self.pitch_high = 1
        self.direction_low = 0
        self.direction_high = 1
        # Action space bounds (4 dims).
        act_low = np.array([self.strength_low, self.frequency_low, self.pitch_low, self.direction_low], dtype=np.float32)
        act_high = np.array([self.strength_high, self.frequency_high, self.pitch_high, self.direction_high], dtype=np.float32)
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

    def reset(self, target_location_millicore=0.5, target_location_nano=0.5, target_release_amount=0, control_points=np.array([[0, 0.5],[0.25, 0.5],[0.5, 0.5],[0.75, 0.5],[1, 0.5]]), millicore_exists=True, nano_exists=True, fix_distance=True):
        """
        Reset episode state.

        Missing robots are encoded with location=-1 and theta=-1.
        """
        x = control_points[:, 0]
        y = control_points[:, 1]

        # Parametric pipe geometry used by dynamics and rendering.
        self.spline = make_interp_spline(x, y, k=3)
        self.x_smooth = np.linspace(x.min(), x.max(), 100)
        self.y_smooth = self.spline(self.x_smooth)

        self.step_number = 0
        self.millicore_exists = millicore_exists
        self.nano_exists = nano_exists
        
        if millicore_exists:
            self.target_location_millicore = target_location_millicore
            self.current_location_millicore = random.uniform(0, 1)
            if fix_distance:
                self.current_location_millicore = random.uniform(self.target_location_millicore - 0.2, self.target_location_millicore + 0.2)
                self.current_location_millicore = min(max(self.current_location_millicore, 0), 1)
        else:
            self.target_location_millicore = -1
            self.current_location_millicore = -1

        if nano_exists:
            self.target_location_nano = target_location_nano
            self.current_location_nano = random.uniform(0, 1)
            if fix_distance:
                self.current_location_nano = random.uniform(self.target_location_nano - 0.2, self.target_location_nano + 0.2)
                self.current_location_nano = min(max(self.current_location_nano, 0), 1)
        else:
            self.target_location_nano = -1
            self.current_location_nano = -1

        # Release control is active only when millicore exists.
        if millicore_exists:
            self.target_release_amount = target_release_amount
            self.current_release_amount = target_release_amount
        else:
            self.target_release_amount = 0
            self.current_release_amount = 0
        
        if self.current_location_millicore != -1 and self.target_location_millicore != -1:
            self.millicore_x = (self.current_location_millicore) * (
                        self.x_smooth.max() - self.x_smooth.min()) + self.x_smooth.min()
            self.millicore_y = self.spline(self.millicore_x)
        else:
            self.millicore_x = None
            self.millicore_y = None

        if self.current_location_nano != -1 and self.target_location_nano != -1:
            self.nanounit_x = (self.current_location_nano) * (
                        self.x_smooth.max() - self.x_smooth.min()) + self.x_smooth.min()
            self.nanounit_y = self.spline(self.nanounit_x)
        else:
            self.nanounit_x = None
            self.nanounit_y = None

        self.calculate_theta()
        if not millicore_exists:
            self.theta_millicore = -1
        if not nano_exists:
            self.theta_nano = -1

        self.total_release_amount = 0

        self.current_obs = np.array([self.target_location_millicore, self.target_location_nano, self.target_release_amount, self.current_location_millicore, self.current_location_nano, self.current_release_amount, self.theta_millicore, self.theta_nano])

        return self.current_obs

    def step(self, action):
        """Advance one environment step and return (obs, reward, done, info)."""
        # Convert a global action into robot-local actions on the pipe tangent frame.
        self.action = action
        self.action_millicore, self.action_nano = self.calculate_action(action)

        action_millicore_tensor = torch.tensor(self.action_millicore, dtype=torch.float32)
        action_nano_tensor = torch.tensor(self.action_nano, dtype=torch.float32)
        with torch.no_grad():
            millicore_input = action_millicore_tensor.unsqueeze(0)
            tail_input = action_nano_tensor.unsqueeze(0)

            sim_velocity_millicore = model_velocity_millicore(millicore_input).detach().numpy() / 2
            sim_velocity_nano = model_velocity_nano(tail_input).detach().numpy()
            sim_gray_scale = model_release_rate(action_millicore_tensor).detach().numpy()

        sim_velocity_millicore = gaussian_noise(sim_velocity_millicore, noise_on=True)
        sim_velocity_nano = gaussian_noise(sim_velocity_nano, noise_on=True)
        sim_gray_scale = gaussian_noise(sim_gray_scale, noise_on=True)

        sim_release_rate = gray_scale2release_rate(sim_gray_scale)
        if action[0] == 0 or action[1] == 0:
            sim_velocity_millicore = 0
            sim_velocity_nano = 0
            sim_release_rate = 0
            
        last_location_millicore = self.current_location_millicore
        last_location_nano = self.current_location_nano

        # Integrate position only for active robots.
        if self.current_location_millicore != -1 and self.target_location_millicore != -1:
            self.current_location_millicore += sim_velocity_millicore
        if self.current_location_nano != -1 and self.target_location_nano != -1:
            self.current_location_nano += sim_velocity_nano
       
        # Aggregate release behavior as a running mean over the episode.
        if self.millicore_exists:
            self.current_release_amount = sim_release_rate
            self.total_release_amount += sim_release_rate
            self.average_release_amount = self.total_release_amount / (self.step_number + 1)
            release_difference = np.abs(self.target_release_amount - self.average_release_amount)
        else:
            self.current_release_amount = 0
            release_difference = 0

        if self.current_location_millicore != -1 and self.target_location_millicore != -1:
            self.millicore_x = (self.current_location_millicore) * (
                        self.x_smooth.max() - self.x_smooth.min()) + self.x_smooth.min()
            self.millicore_y = self.spline(self.millicore_x)
        else:
            self.millicore_x = None
            self.millicore_y = None

        if self.current_location_nano != -1 and self.target_location_nano != -1:
            self.nanounit_x = (self.current_location_nano) * (
                        self.x_smooth.max() - self.x_smooth.min()) + self.x_smooth.min()
            self.nanounit_y = self.spline(self.nanounit_x)
        else:
            self.nanounit_x = None
            self.nanounit_y = None

        self.calculate_theta()

        # Observation vector follows self.observation_space layout.
        self.current_obs = np.array([float(self.target_location_millicore), float(self.target_location_nano), float(self.target_release_amount), float(self.current_location_millicore), float(self.current_location_nano), float(self.current_release_amount), float(self.theta_millicore), float(self.theta_nano)]).squeeze()

        # Per-task shaping terms.
        reward_location_millicore = 0
        reward_location_nano = 0
        reward_release = 0
        if self.target_location_millicore != -1:
            reward_location_millicore = -np.abs(self.current_location_millicore - self.target_location_millicore)
            
        if self.target_location_nano != -1:
            reward_location_nano = -np.abs(self.current_location_nano - self.target_location_nano)
        
        if self.millicore_exists:
            reward_release = -np.abs(self.target_release_amount-self.current_release_amount)

        reward_distance = -(reward_location_millicore**2 + reward_location_nano**2 + reward_release**2)**0.5

        reward = 100 * (-0.1 + reward_distance)

        done = False

        if self.step_number >= 100:
            done = True

        # Out-of-bound termination for active robots.
        if self.current_location_millicore != -1:
            if self.current_location_millicore >= 1.5 or self.current_location_millicore <= -0.5:
                done = True
                reward -= 10 * (100 - self.step_number)
                print('Millicore out of bounds')

        if self.current_location_nano != -1:
            if self.current_location_nano >= 1.5 or self.current_location_nano <= -0.5:
                done = True
                reward -= 10 * (100 - self.step_number)
                print('Nanounits out of bounds')

        # Task completion checks.
        millicore_reached = False
        nano_reached = False
        if self.current_location_millicore != -1 and self.target_location_millicore != -1:
            # Direction-aware threshold to avoid sign ambiguity near the target.
            if self.step_number > 0:
                if last_location_millicore < self.target_location_millicore:
                    millicore_reached = (self.target_location_millicore - self.current_location_millicore) <= 0.05
                else:
                    millicore_reached = (self.target_location_millicore - self.current_location_millicore) >= -0.05
            else:
                millicore_reached = np.abs(self.current_location_millicore - self.target_location_millicore) <= 0.05
        elif self.current_location_millicore == -1 and self.target_location_millicore == -1:
            millicore_reached = True
            
        if self.current_location_nano != -1 and self.target_location_nano != -1:
            if self.step_number > 0:
                if last_location_nano < self.target_location_nano:
                    nano_reached = (self.target_location_nano - self.current_location_nano) <= 0.05
                else:
                    nano_reached = (self.target_location_nano - self.current_location_nano) >= -0.05
            else:
                nano_reached = np.abs(self.current_location_nano - self.target_location_nano) <= 0.05
        elif self.current_location_nano == -1 and self.target_location_nano == -1:
            nano_reached = True

        # Release criterion is evaluated only when release control is active.
        release_correct = True
        if self.millicore_exists:
            threshold = self.release_thresholds[self.target_release_amount]
            if release_difference > threshold:
                release_correct = False

        if millicore_reached and nano_reached:
            done = True
            reward += 1000
            print('Millicore and nanounits reached target location!')
        elif done and not millicore_reached and nano_reached:
            print('nano reached target location, millicore not')
        elif done and millicore_reached and not nano_reached:
            print('millicore reached target location, nano not')

        if done and release_correct:
            reward += 1000
            print('release amount correct')

        if done and not release_correct:
            print(f'average_release_amount: {float(self.average_release_amount):.3f}, release_difference: {float(release_difference):.3f}')

            
        if release_correct and millicore_reached and nano_reached:
                reward += 1000
                print('All tasks completed!!!!!')

        if done:
            reward -= 10 * self.step_number

        # Episode-end task errors for logging/analysis.
        millicore_error = None
        nano_error = None
        release_error = None
        
        if self.millicore_exists and self.target_location_millicore != -1:
            millicore_error = float(np.abs(self.current_location_millicore - self.target_location_millicore))
        
        if self.nano_exists and self.target_location_nano != -1:
            nano_error = float(np.abs(self.current_location_nano - self.target_location_nano))
        
        if self.millicore_exists:
            release_error = float(release_difference)
        
        # Structured diagnostics for training logs and post hoc analysis.
        info = {
            'millicore_reached': millicore_reached,
            'nano_reached': nano_reached,
            'release_correct': release_correct,
            'all_tasks_completed': millicore_reached and nano_reached and release_correct,
            'millicore_error': millicore_error,
            'nano_error': nano_error,
            'release_error': release_error,
        }

        self.step_number += 1
        return self.current_obs, reward, done, info

    def render(self, mode='human', save_video=False, video_name='render.mp4', close=False):
        if close:
            cv2.destroyAllWindows()
            return

        # Canvas size in pixels.
        height = 800
        width = 800

        # Lazily create a writer when video recording is enabled.
        if save_video and not hasattr(self, 'video_writer'):
            video_dir = os.path.join('tmp', 'video')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, video_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (height, width))

        window_name = 'Custom Pipe Visualization with Robots'
        cv2.namedWindow(window_name)

        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        overlay = img.copy()

        for i in range(len(self.x_smooth) - 1):
            pt1 = (int(self.x_smooth[i] * width), int((1 - self.y_smooth[i]) * height))
            pt2 = (int(self.x_smooth[i + 1] * width), int((1 - self.y_smooth[i + 1]) * height))
            cv2.line(overlay, pt1, pt2, (255, 0, 0), 30)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if self.millicore_exists:
            if np.abs(self.target_release_amount-self.current_release_amount) <= 0.1:
                cv2.putText(img, f'Correct release', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)


        if self.target_location_millicore != -1:
            target_millicore_x = (self.target_location_millicore) * (
                        self.x_smooth.max() - self.x_smooth.min()) + self.x_smooth.min()
            target_millicore_y = self.spline(target_millicore_x)
            target_millicore_x_pixel = int(target_millicore_x * width)
            target_millicore_y_pixel = int((1 - target_millicore_y) * height)
            cv2.circle(img, (target_millicore_x_pixel, target_millicore_y_pixel), 5, (0, 255, 0), -1)
            cv2.putText(img, f'Target millicore: {float(self.target_location_millicore):.2f}',
                        (width-450, height-300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f'Target millicore: N/A',
                        (width-450, height-300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        if self.target_location_nano != -1:
            target_nano_x = (self.target_location_nano) * (
                        self.x_smooth.max() - self.x_smooth.min()) + self.x_smooth.min()
            target_nano_y = self.spline(target_nano_x)
            target_nano_x_pixel = int(target_nano_x * width)
            target_nano_y_pixel = int((1 - target_nano_y) * height)
            cv2.circle(img, (target_nano_x_pixel, target_nano_y_pixel), 5, (255, 255, 255), -1)
            cv2.putText(img, f'Target nanounits: {float(self.target_location_nano):.2f}',
                        (width-450, height-200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f'Target nanounits: N/A',
                        (width-450, height-200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        if self.current_location_millicore != -1 and self.target_location_millicore != -1:
            millicore_x_pixel = int(self.millicore_x * width)
            millicore_y_pixel = int((1 - self.millicore_y) * height)
        else:
            millicore_x_pixel = None
            millicore_y_pixel = None
            
        if self.current_location_nano != -1 and self.target_location_nano != -1:
            nanounit_x_pixel = int(self.nanounit_x * width)
            nanounit_y_pixel = int((1 - self.nanounit_y) * height)
        else:
            nanounit_x_pixel = None
            nanounit_y_pixel = None

        # Show heading vectors in image coordinates.
        arrow_length = 50
        
        if millicore_x_pixel is not None:
            arrow_x_millicore = int(millicore_x_pixel + arrow_length * np.cos(self.theta_millicore))
            arrow_y_millicore = int(millicore_y_pixel + arrow_length * np.sin(self.theta_millicore))

            print('action', self.action)
            print('action 3 ', self.action[3]*2*3.1416)
            arrow_x_direction_global = int(millicore_x_pixel + arrow_length * np.cos(self.action[3]*2*3.1416))
            arrow_y_direction_global = int(millicore_y_pixel + arrow_length * np.sin(self.action[3]*2*3.1416))
            cv2.arrowedLine(img, (millicore_x_pixel, millicore_y_pixel), (arrow_x_direction_global, arrow_y_direction_global), (255, 255, 0), 2)

            cv2.arrowedLine(img, (millicore_x_pixel, millicore_y_pixel), (arrow_x_millicore, arrow_y_millicore), (0, 255, 0), 2)

            # `millicore.png` is expected to live in `./utils/` relative to this file.
            img_path = 'utils/millicore.png'
            robot_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            scale = 0.15
            angle = float(-self.theta_millicore * 180 / np.pi)

            new_width = int(robot_img.shape[1] * scale)
            new_height = int(robot_img.shape[0] * scale)
            resized_img = cv2.resize(robot_img, (new_width, new_height))

            center = (new_width // 2, new_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(resized_img, rotation_matrix, (new_width, new_height))

            top_left_x = millicore_x_pixel - new_width // 2
            top_left_y = millicore_y_pixel - new_height // 2

            top_left_x = max(0, min(top_left_x, img.shape[1] - new_width))
            top_left_y = max(0, min(top_left_y, img.shape[0] - new_height))

            for c in range(0, 3):
                img[top_left_y:top_left_y + new_height, top_left_x:top_left_x + new_width, c] = \
                    rotated_img[:, :, c] * (rotated_img[:, :, 3] / 255.0) + \
                    img[top_left_y:top_left_y + new_height, top_left_x:top_left_x + new_width, c] * (
                                1 - (rotated_img[:, :, 3] / 255.0))

        if nanounit_x_pixel is not None:
            arrow_x_nano = int(nanounit_x_pixel + arrow_length * np.cos(self.theta_nano))
            arrow_y_nano = int(nanounit_y_pixel + arrow_length * np.sin(self.theta_nano))

            arrow_x_direction_global = int(nanounit_x_pixel + arrow_length * np.cos(self.action[3]*2*3.1416))
            arrow_y_direction_global = int(nanounit_y_pixel + arrow_length * np.sin(self.action[3]*2*3.1416))
            cv2.arrowedLine(img, (nanounit_x_pixel, nanounit_y_pixel), (arrow_x_direction_global, arrow_y_direction_global), (255, 255, 0), 2)

            cv2.arrowedLine(img, (nanounit_x_pixel, nanounit_y_pixel), (arrow_x_nano, arrow_y_nano), (255, 255, 255), 2)

        # Display global and robot-frame actions in physical units.
        cv2.putText(img, f'Global action: {np.round(self.action*[20, 40, 180, 360], 2)}',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'Millicore action: {np.round(self.action_millicore*[20, 40, 180, 360], 2)}',
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'Nano action: {np.round(self.action_nano*[20, 40, 180, 360], 2)}',
                    (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)



        if millicore_x_pixel is not None:
            cv2.circle(img, (millicore_x_pixel, millicore_y_pixel), 3, (255, 0, 0), -1)
            cv2.putText(img, f'millicore: {float(self.current_location_millicore):.2f}',
                        (width-450, height-250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            if self.theta_millicore != -1:
                cv2.putText(img, f'Theta millicore: {float(self.theta_millicore*180/3.1416):.2f}',
                            (width-750, height-250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, f'Theta millicore: N/A',
                            (width-750, height-250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f'millicore: N/A',
                        (width-450, height-250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        if nanounit_x_pixel is not None:
            cv2.circle(img, (nanounit_x_pixel, nanounit_y_pixel), 10, (0, 0, 255), -1)
            cv2.putText(img, f'nanounits: {float(self.current_location_nano):.2f}',
                        (width-450, height-150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            if self.theta_nano != -1:
                cv2.putText(img, f'Theta nanounits: {float(self.theta_nano*180/3.1416):.2f}',
                            (width-750, height-150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, f'Theta nanounits: N/A',
                            (width-750, height-150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f'nanounits: N/A',
                        (width-450, height-150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)




        if self.millicore_exists:
            cv2.putText(img, f'Target release amount: {float(self.target_release_amount):.2f}',
                        (width-450, height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f'Current release amount: {float(self.current_release_amount):.2f}',
                        (width-450, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f'Average release amount: {float(self.average_release_amount):.2f}',
                        (width-450, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f'No release behavior',
                        (width-450, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # Release indicator panel.
        if self.millicore_exists:
            color_bar_height = 50
            color_bar_width = 100
            current_amount = float(self.current_release_amount)

            color_value = int((1 - current_amount) * 255)
            color = (color_value, color_value, color_value)

            top_left_x = width - color_bar_width - 100
            top_left_y = 10
            bottom_right_x = top_left_x + color_bar_width
            bottom_right_y = top_left_y + color_bar_height

            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, -1)
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)

            cv2.putText(img, f'Current release rate: {float(self.current_release_amount):.2f}',
                        (top_left_x-150, top_left_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(window_name, img)

        if save_video and hasattr(self, 'video_writer'):
            self.video_writer.write(img)

        cv2.waitKey(1)

    def calculate_theta(self):
        # Finite-difference estimate of local tangent direction.
        delta = 1e-4

        if self.target_location_millicore != -1 and self.current_location_millicore != -1:
            millicore_y_plus = self.spline(self.millicore_x + delta)
            millicore_y_minus = self.spline(self.millicore_x - delta)
            dx_millicore = 2 * delta
            dy_millicore = millicore_y_plus - millicore_y_minus

            if dy_millicore >= 0:
                if self.current_location_millicore <= self.target_location_millicore:
                    self.theta_millicore = -np.arctan2(dy_millicore, dx_millicore)
                else:
                    self.theta_millicore = -(np.arctan2(dy_millicore, dx_millicore) + np.pi)
            else:
                if self.current_location_millicore <= self.target_location_millicore:
                    self.theta_millicore = -(2 * np.pi + np.arctan2(dy_millicore, dx_millicore))
                else:
                    self.theta_millicore = -(np.pi + np.arctan2(dy_millicore, dx_millicore))

            self.theta_millicore = (self.theta_millicore + 8 * np.pi) % (2*np.pi)
        else:
            self.theta_millicore = -1

        if self.target_location_nano != -1 and self.current_location_nano != -1:
            nanounit_y_plus = self.spline(self.nanounit_x + delta)
            nanounit_y_minus = self.spline(self.nanounit_x - delta)
            dx_nano = 2 * delta
            dy_nano = nanounit_y_plus - nanounit_y_minus

            if dy_nano >= 0:
                if self.current_location_nano <= self.target_location_nano:
                    self.theta_nano = -(np.arctan2(dy_nano, dx_nano))
                else:
                    self.theta_nano = -(np.arctan2(dy_nano, dx_nano) + np.pi)
            else:
                if self.current_location_nano <= self.target_location_nano:
                    self.theta_nano = -(2 * np.pi + np.arctan2(dy_nano, dx_nano))
                else:
                    self.theta_nano = -(np.pi + np.arctan2(dy_nano, dx_nano))

            self.theta_nano = (self.theta_nano + 8 * np.pi) % (2 * np.pi)
        else:
            self.theta_nano = -1

    def calculate_action(self, action):
        # Map global direction command to each robot's local tangent frame.
        if self.theta_millicore != -1:
            if (self.theta_millicore >= 0) and (self.theta_millicore < np.pi/2):
                angle_millicore = self.theta_millicore
            elif (self.theta_millicore >= np.pi/2) and (self.theta_millicore < np.pi):
                angle_millicore = self.theta_millicore - np.pi
            elif (self.theta_millicore >= np.pi) and (self.theta_millicore < 3/2*np.pi):
                angle_millicore = self.theta_millicore - np.pi
            elif (self.theta_millicore >= 3/2*np.pi) and (self.theta_millicore < 2*np.pi):
                angle_millicore = self.theta_millicore - 2*np.pi
        else:
            angle_millicore = 0

        if self.theta_nano != -1:
            if (self.theta_nano >= 0) and (self.theta_nano < np.pi / 2):
                angle_nano = self.theta_nano
            elif (self.theta_nano >= np.pi / 2) and (self.theta_nano < np.pi):
                angle_nano = self.theta_nano - np.pi
            elif (self.theta_nano >= np.pi) and (self.theta_nano < 3 / 2 * np.pi):
                angle_nano = self.theta_nano - np.pi
            elif (self.theta_nano >= 3 / 2 * np.pi) and (self.theta_nano < 2 * np.pi):
                angle_nano = self.theta_nano - 2*np.pi
        else:
            angle_nano = 0

        action_millicore = action.copy()
        action_nano = action.copy()

        global_direction = action.copy()[3] * 2 * np.pi

        millicore_direction = ((global_direction - angle_millicore) + (8 * np.pi)) % (2 * np.pi)
        nano_direction = ((global_direction - angle_nano) + (8 * np.pi)) % (2 * np.pi)

        action_millicore[3] = millicore_direction / (2*np.pi)
        action_nano[3] = nano_direction / (2*np.pi)

        return action_millicore, action_nano

    def close_video(self):
        """Release the video writer handle if it exists."""
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            del self.video_writer

    def close(self):
        """
        Optional cleanup hook.
        """
        pass

    def get_release_thresholds(self):
        """Return release-task tolerances by target class."""
        return self.release_thresholds

    def seed(self, seed=None):
        """
        Optional Gym-compatible seed hook.
        """
        return
