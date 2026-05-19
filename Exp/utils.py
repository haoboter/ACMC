import cv2
import numpy as np
import os
from scipy.interpolate import make_interp_spline

def calculate_curve_length(x_smooth, y_smooth):
    """Compute arc length of curve."""
    length = 0.0
    for i in range(1, len(x_smooth)):
        segment_length = np.sqrt((x_smooth[i] - x_smooth[i - 1])**2 + (y_smooth[i] - y_smooth[i - 1])**2)
        length += segment_length
    return length

def median_filter(data, window_size):
    """Median filter for trajectory smoothing (endpoints preserved)."""
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size // 2 or i >= len(data) - window_size // 2:
            smoothed_data.append(data[i])
        else:
            smoothed_data.append(np.median(data[i - window_size // 2:i + window_size // 2 + 1], axis=0))
    return smoothed_data

def moving_average(data, window_size):
    """Moving-average smoothing with fixed window size."""
    if len(data) < window_size:
        return data
    return [np.mean(data[i:i + window_size], axis=0) for i in range(len(data) - window_size + 1)]

def find_nearest_point(millicore_coord, x_smooth, y_smooth):
    """Project a point onto the curve and compute tangent direction + normalized arc-length."""
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

def gray_scale2release_rate(gray_scale):
    """Map grayscale intensity to release rate in {0, 1}."""
    release_rate = 0
    if gray_scale:
        if gray_scale <= 60:
            release_rate = 1
        elif gray_scale >= 180:
            release_rate = 0
        else:
            release_rate = 1-(np.abs(gray_scale - 50) / 120)
    return release_rate

def save_selected_points(normalized_points, filename='tmp/selected_points.npy'):
    """Save selected pipe control points."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, normalized_points)
    print(f"Selected points saved to: {filename}")

def load_selected_points(filename='tmp/selected_points.npy'):
    """Load selected pipe control points from disk."""
    if os.path.exists(filename):
        normalized_points = np.load(filename)
        print(f"Loaded selected points from: {filename}")
        return normalized_points
    else:
        print(f"File not found: {filename}")
        return None

def select_points(frame, load_from_file=False, save_file='tmp/selected_points.npy'):
    """
    Select pipe control points on a frame and fit a spline.
    Returns: (x_smooth, y_smooth, normalized_points) or None if cancelled.
    """
    if load_from_file:
        normalized_points_loaded = load_selected_points(save_file)
        if normalized_points_loaded is not None and len(normalized_points_loaded) >= 2:
            x = normalized_points_loaded[:, 0]
            y = normalized_points_loaded[:, 1]
            spline = make_interp_spline(x, y, k=3)
            x_smooth = np.linspace(x.min(), x.max(), 500)
            y_smooth = spline(x_smooth)
            print("Loaded points are valid; using them to fit spline.")
            return np.array(x_smooth), np.array(y_smooth), normalized_points_loaded
        else:
            print("Loaded points are invalid; please select again.")

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
        print("Press 'Enter' to confirm, press 'ESC' to reselect.")
        key = cv2.waitKey(0)

        if key != 13:
            return None

        cv2.destroyAllWindows()
        
        save_selected_points(np.array(normalized_points), save_file)

    print("Selected points:", normalized_points)

    return np.array(x_smooth), np.array(y_smooth), np.array(normalized_points)