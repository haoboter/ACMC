"""Architectures and helpers for SL-trained digital twins (behavior predictors used in ``sim_env``),
plus plotting utilities and interactive pipe point picking."""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_score_history(mean_score_history, save_path, window_size=50):
    """
    Plot average score history.
    
    Args:
        mean_score_history: List of average scores over episodes.
        save_path: Path for saving the figure.
        window_size: Window size used for computing the average (for title display).
    """
    plt.figure(figsize=(10, 6))
    num_scores = len(mean_score_history)
    x = np.linspace(0, num_scores - 1, num_scores)
    plt.plot(x, mean_score_history, marker='o', linestyle='-', color='b', label='Average score')
    plt.title(f'Average score over {window_size} episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.xlim(0, num_scores + 1)
    plt.ylim(np.min(mean_score_history)-10, np.max(mean_score_history)+10)
    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def gaussian_noise(signal, noise_on = True, mean=1.0, std_dev=0.1):
    """Multiplicative perturbation: ``signal * N(mean, std)``; mean≈1 keeps typical scale."""
    if noise_on:
        gaussian_noise = np.random.normal(mean, std_dev, size=signal.shape)
        noised_signal = signal * gaussian_noise
    else:
        noised_signal = signal

    return noised_signal

class NN_motion_prediction(nn.Module):
    def __init__(self, input_dim=4):
        super(NN_motion_prediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=True)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, track_running_stats=True)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, track_running_stats=True)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x))) 

        output = torch.tanh(self.fc4(x))
        return output

class NN_grayscale_prediction(nn.Module):
    def __init__(self, input_dim=4):
        super(NN_grayscale_prediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=True)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256, track_running_stats=True)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, track_running_stats=True)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x))) 

        output = torch.sigmoid(self.fc4(x)) * 255.0
        return output


def gray_scale2release_rate(gray_scale):
    """Piecewise map from the release digital twin's grayscale output to release rate in [0, 1] (hinges at 50 and 200)."""
    release_rate = 0
    if gray_scale:
        if gray_scale <= 50:
            release_rate = 1
        elif gray_scale >= 200:
            release_rate = 0
        else:
            release_rate = 1-(np.abs(gray_scale - 50) / 150)
    return release_rate

def select_points(frame):
    points = []
    frame_copy = frame.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select the pipe', frame_copy)

    cv2.imshow('Select the pipe', frame_copy)
    cv2.setMouseCallback('Select the pipe', click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key == 13:
            break

    normalized_points = [(x / frame.shape[1], y / frame.shape[0]) for (x, y) in points]
    if len(points) >= 2:
        x = np.array([p[0] for p in normalized_points])
        y = np.array([p[1] for p in normalized_points])
        spline = make_interp_spline(x, y, k=3)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = spline(x_smooth)


        for i in range(len(x_smooth) - 1):
            cv2.line(frame_copy, (int(x_smooth[i]*frame.shape[1]), int(y_smooth[i]*frame.shape[0])), (int(x_smooth[i + 1]*frame.shape[1]), int(y_smooth[i + 1]*frame.shape[0])), (255, 0, 0), 2)

        cv2.imshow('Camera Feed with Curve', frame_copy)
        print("Press 'Enter' to comfirm, press 'ESC' to reselect")
        key = cv2.waitKey(0)

        if key != 13:
            return None

        cv2.destroyAllWindows()

    print("Selected points:", normalized_points)

    return np.array(x_smooth), np.array(y_smooth), np.array(normalized_points)


def plot_task_completion_rates(millicore_completion_history, nano_completion_history, 
                                release_completion_history, all_tasks_completion_history,
                                save_path, window_size=50, task_category_completion_history=None, 
                                task_category_episode_numbers=None, current_episode=None):
    """
    Plot task completion rates.
    
    Args:
        millicore_completion_history: Completion history for the millicore task.
        nano_completion_history: Completion history for the nanounits task.
        release_completion_history: Completion history for the release task.
        all_tasks_completion_history: History of simultaneous completion of all tasks.
        save_path: Path for saving the figure.
        window_size: Sliding-window size, default is 50.
        task_category_completion_history: Category-wise completion history dict, e.g. {'task_a': [0,1,0,...], ...}
        task_category_episode_numbers: Category-wise episode number dict, e.g. {'task_a': [1,5,10,...], ...}
        current_episode: Current episode index.
    
    Returns:
        tuple: (millicore_success_rate, nano_success_rate, release_success_rate, all_tasks_success_rate)
    """
    millicore_success_rate = float(np.mean(millicore_completion_history[-window_size:]))
    nano_success_rate = float(np.mean(nano_completion_history[-window_size:]))
    release_success_rate = float(np.mean(release_completion_history[-window_size:]))
    all_tasks_success_rate = float(np.mean(all_tasks_completion_history[-window_size:]))
    
    fig = plt.figure(figsize=(16, 12))
    
    millicore_rates = []
    nano_rates = []
    release_rates = []
    all_tasks_rates = []
    
    for j in range(window_size - 1, len(millicore_completion_history)):
        start_idx = j - window_size + 1
        end_idx = j + 1
        millicore_rates.append(float(np.mean(millicore_completion_history[start_idx:end_idx])))
        nano_rates.append(float(np.mean(nano_completion_history[start_idx:end_idx])))
        release_rates.append(float(np.mean(release_completion_history[start_idx:end_idx])))
        all_tasks_rates.append(float(np.mean(all_tasks_completion_history[start_idx:end_idx])))
    
    episodes = list(range(window_size, window_size + len(millicore_rates)))
    
    plt.subplot(3, 1, 1)
    plt.plot(episodes, millicore_rates, label='Millicore Task Completion Rate', color='red', linewidth=2)
    plt.plot(episodes, nano_rates, label='Nanounits Task Completion Rate', color='blue', linewidth=2)
    plt.plot(episodes, release_rates, label='Release Task Completion Rate', color='green', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Completion Rate (Last 50 Episodes)', fontsize=12)
    plt.title('Individual Task Completion Rates', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    plt.subplot(3, 1, 2)
    plt.plot(episodes, all_tasks_rates, label='All Tasks Simultaneous Completion Rate', color='purple', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Completion Rate (Last 50 Episodes)', fontsize=12)
    plt.title('All Tasks Simultaneous Completion Rate', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    if task_category_completion_history and task_category_episode_numbers and current_episode is not None:
        plt.subplot(3, 1, 3)
        
        task_labels = {
            'task_a': 'Task A: release=0, millicore=True, nano=False',
            'task_b': 'Task B: release=1, millicore=True, nano=False',
            'task_c': 'Task C: millicore=False, nano=True (no release)',
            'task_e': 'Task E: release=0, millicore=True, nano=True',
            'task_f': 'Task F: release=1, millicore=True, nano=True'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#98D8C8', '#F7DC6F']
        
        start_episode = 50
        
        if current_episode - start_episode > 1000:
            step = 10
            all_episodes = list(range(start_episode, current_episode + 1, step))
        else:
            all_episodes = list(range(start_episode, current_episode + 1))
        
        for idx, (task_key, color) in enumerate(zip(['task_a', 'task_b', 'task_c', 'task_e', 'task_f'], colors)):
            history = task_category_completion_history[task_key]
            episode_numbers = task_category_episode_numbers[task_key]
            
            if len(history) > 0 and len(episode_numbers) == len(history):
                episode_numbers_arr = np.array(episode_numbers)
                history_arr = np.array(history)
                
                sliding_window_rates = []
                
                for ep in all_episodes:
                    mask = episode_numbers_arr <= ep
                    experiments_before_ep = history_arr[mask]
                    
                    if len(experiments_before_ep) >= window_size:
                        recent_completions = experiments_before_ep[-window_size:]
                        rate = float(np.mean(recent_completions))
                    elif len(experiments_before_ep) > 0:
                        rate = float(np.mean(experiments_before_ep))
                    else:
                        rate = float('nan')
                    
                    sliding_window_rates.append(rate)
                
                plt.plot(all_episodes, sliding_window_rates, label=task_labels[task_key], 
                        color=color, linewidth=2, alpha=0.7)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(f'Completion Rate (Last {window_size} Experiments)', fontsize=12)
        plt.title('Task Category Completion Rates (Sliding Window)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return millicore_success_rate, nano_success_rate, release_success_rate, all_tasks_success_rate


def plot_task_errors(millicore_error_history, nano_error_history, release_error_history,
                     save_path, window_size=50):
    """
    Plot task errors using sliding-window averages.
    
    Args:
        millicore_error_history: Error history for the millicore task (None means the task does not exist).
        nano_error_history: Error history for the nanounits task (None means the task does not exist).
        release_error_history: Error history for the release task (None means the task does not exist).
        save_path: Path for saving the figure.
        window_size: Sliding-window size, default is 50.
    
    Returns:
        tuple: (millicore_avg_error, nano_avg_error, release_avg_error, rmse_avg_error)
    """
    
    def compute_sliding_avg_with_none(error_list, window_size):
        """Compute a sliding average while skipping None values."""
        avg_errors = []
        for j in range(window_size - 1, len(error_list)):
            start_idx = j - window_size + 1
            end_idx = j + 1
            window_data = error_list[start_idx:end_idx]
            valid_data = [x for x in window_data if x is not None]
            if len(valid_data) > 0:
                avg_errors.append(float(np.mean(valid_data)))
            else:
                avg_errors.append(float('nan'))
        return avg_errors
    
    def compute_sliding_rmse(millicore_list, nano_list, release_list, window_size):
        """
        Compute sliding-window RMSE across available task errors.
        """
        rmse_errors = []
        for j in range(window_size - 1, len(millicore_list)):
            start_idx = j - window_size + 1
            end_idx = j + 1
            
            window_millicore = millicore_list[start_idx:end_idx]
            window_nano = nano_list[start_idx:end_idx]
            window_release = release_list[start_idx:end_idx]
            
            episode_rmses = []
            for k in range(len(window_millicore)):
                errors = []
                if window_millicore[k] is not None:
                    errors.append(window_millicore[k])
                if window_nano[k] is not None:
                    errors.append(window_nano[k])
                if window_release[k] is not None:
                    errors.append(window_release[k])
                
                if len(errors) > 0:
                    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
                    episode_rmses.append(rmse)
            
            if len(episode_rmses) > 0:
                rmse_errors.append(float(np.mean(episode_rmses)))
            else:
                rmse_errors.append(float('nan'))
        
        return rmse_errors
    
    millicore_avg_errors = compute_sliding_avg_with_none(millicore_error_history, window_size)
    nano_avg_errors = compute_sliding_avg_with_none(nano_error_history, window_size)
    release_avg_errors = compute_sliding_avg_with_none(release_error_history, window_size)
    rmse_avg_errors = compute_sliding_rmse(millicore_error_history, nano_error_history, 
                                           release_error_history, window_size)
    
    # Histories may have different lengths (e.g., some tasks absent for long stretches).
    # Always align x/y by trimming all series to the same length.
    n = min(
        len(millicore_avg_errors),
        len(nano_avg_errors),
        len(release_avg_errors),
        len(rmse_avg_errors),
    )
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    millicore_avg_errors = millicore_avg_errors[-n:]
    nano_avg_errors = nano_avg_errors[-n:]
    release_avg_errors = release_avg_errors[-n:]
    rmse_avg_errors = rmse_avg_errors[-n:]
    episodes = list(range(window_size, window_size + n))
    
    def get_recent_avg(error_list):
        recent = error_list[-window_size:]
        valid = [x for x in recent if x is not None]
        return float(np.mean(valid)) if len(valid) > 0 else float('nan')
    
    millicore_recent_avg = get_recent_avg(millicore_error_history)
    nano_recent_avg = get_recent_avg(nano_error_history)
    release_recent_avg = get_recent_avg(release_error_history)
    rmse_recent_avg = rmse_avg_errors[-1] if len(rmse_avg_errors) > 0 else float('nan')
    
    fig = plt.figure(figsize=(16, 9))
    
    plt.subplot(2, 1, 1)
    plt.plot(episodes, millicore_avg_errors, label='Millicore Position Error', color='red', linewidth=2)
    plt.plot(episodes, nano_avg_errors, label='Nanounits Position Error', color='blue', linewidth=2)
    plt.plot(episodes, release_avg_errors, label='Release Amount Error', color='green', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Average Error (Last {window_size} Episodes)', fontsize=12)
    plt.title('Individual Task Errors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(episodes, rmse_avg_errors, label='Combined RMSE (All Active Tasks)', 
             color='purple', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'RMSE (Last {window_size} Episodes)', fontsize=12)
    plt.title('Combined Root Mean Square Error', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return millicore_recent_avg, nano_recent_avg, release_recent_avg, rmse_recent_avg