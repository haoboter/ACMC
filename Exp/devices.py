import matplotlib
matplotlib.use('Agg')
import gxipy as gx
import numpy as np
import cv2
import os
import time
import S826
import torch
import pandas as pd
from utils import *
import threading
global model
global model_yolo
from ultralytics import YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\n\nDevice Used:', device)
# Initialization
model_yolo = YOLO("best.pt", verbose=False)
model_yolo.to(device)

class Coil:
    def __init__(self):
        self.coil = S826.S826()
        self.flux = 0
        self.frequency = 0
        self.pitch = 0
        self.direction = 0

        self.state = True
        self.thread = threading.Thread(target=self.rotating_field)
        self.thread.start()

    def rotating_field(self):
        while self.state:
            self.rotating_field_signal()
            time.sleep(1e-7)

    def rotating_field_signal(self):
        self.coil.rotating_field(flux_density=self.flux, frequency=self.frequency, pitch=self.pitch, direction=self.direction)

    def update_field(self, new_flux, new_freq, new_pitch, new_direct):
        self.flux = new_flux
        self.frequency = new_freq
        self.pitch = new_pitch
        self.direction = new_direct

    def stop(self):
        self.update_field(0, 0, 0, 0)
        time.sleep(0.1)
        self.state = False
        self.thread.join()


class Camera:
    def __init__(self):
        self.images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_images')
        self.cam = None

    def camera_setting(self):
        self.device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.device_manager.update_device_list()
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            return

        self.cam = self.device_manager.open_device_by_index(1)

        self.cam.Width.set(2448)
        self.cam.Height.set(2048)
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        self.cam.ExposureTime.set(8000)
        self.cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
        self.cam.Gain.set(1)
        self.cam.UserSetSelector.set(1)
        self.cam.UserSetSave.send_command()

        self.gamma_lut = gx.Utility.get_gamma_lut(
            self.cam.GammaParam.get()) if self.cam.GammaParam.is_readable() else None
        self.contrast_lut = gx.Utility.get_contrast_lut(
            self.cam.ContrastParam.get()) if self.cam.ContrastParam.is_readable() else None
        self.color_correction_param = self.cam.ColorCorrectionParam.get() if self.cam.ColorCorrectionParam.is_readable() else 0

        self.cam.data_stream[0].set_acquisition_buffer_number(2)
        self.cam.stream_on()
        
        print("Waiting for white balance to stabilize...")
        for i in range(5):
            raw_image = self.cam.data_stream[0].get_image()
            if raw_image is not None:
                time.sleep(0.1)
        print("White balance stabilized.")

    def get_image_init(self):
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed.")
            return

        rgb_image = raw_image.convert("RGB")
        rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)
        numpy_image = rgb_image.get_numpy_array()
        frame = cv2.cvtColor(np.asarray(numpy_image), cv2.COLOR_RGB2BGR)
        resized_frame = resize_image(frame, scale_factor=0.25)

        return resized_frame

    def get_image(self):
        raw_image = self.cam.data_stream[0].get_image()

        rgb_image = raw_image.convert("RGB")
        rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)
        numpy_image = rgb_image.get_numpy_array()
        frame = cv2.cvtColor(np.asarray(numpy_image), cv2.COLOR_RGB2BGR)
        resized_frame = resize_image(frame, scale_factor=0.25)

        return resized_frame

    def get_init_info(self, pipe_width=0.1, circle_diameter=0.1, threshold_nano=100):
        self.threshold_nano = threshold_nano
        origin_frame = self.get_image_init()
        init_frame = origin_frame.copy()

        save_file_1 = 'tmp/selected_points_path1.npy'
        save_file_2 = 'tmp/selected_points_path2.npy'
        
        load_from_file_1 = False
        load_from_file_2 = False
        
        if os.path.exists(save_file_1) or os.path.exists(save_file_2):
            user_input = input("Load previously selected points? (y/n, default n): ").strip().lower()
            if user_input == 'y' or user_input == 'yes':
                if os.path.exists(save_file_1):
                    load_from_file_1 = True
                if os.path.exists(save_file_2):
                    load_from_file_2 = True
        
        self.x_smooth, self.y_smooth, normalized_pipe_points = select_points(init_frame, load_from_file=load_from_file_1, save_file=save_file_1)
        
        if normalized_pipe_points is None:
            print("Point selection was cancelled.")
            return None

        self.spline = make_interp_spline(normalized_pipe_points[:, 0], normalized_pipe_points[:, 1], k=3)

        pipe_mask_rgb = draw_pipe_mask(init_frame, self.x_smooth, self.y_smooth, pipe_width=pipe_width)
        self.pipe_mask_1 = cv2.cvtColor(pipe_mask_rgb, cv2.COLOR_BGR2GRAY)
        pipe_mask_rgb_yolo = draw_pipe_mask(init_frame, self.x_smooth, self.y_smooth, pipe_width=pipe_width * 2)
        self.pipe_mask_yolo_1 = cv2.cvtColor(pipe_mask_rgb_yolo, cv2.COLOR_BGR2GRAY)
        self.pipe_mask = self.pipe_mask_1.copy()
        self.pipe_mask_yolo = self.pipe_mask_yolo_1.copy()

        self.x_smooth_2 = []
        self.y_smooth_2 = []
        self.spline_2 = []
        self.pipe_mask_2 = None
        self.pipe_mask_yolo_2 = None
        if True:
            self.x_smooth_2, self.y_smooth_2, normalized_pipe_points_2 = select_points(
                init_frame, load_from_file=load_from_file_2, save_file=save_file_2
            )
            
            if normalized_pipe_points_2 is None:
                print("Point selection was cancelled.")
                return None
                
            self.spline_2 = make_interp_spline(normalized_pipe_points_2[:, 0], normalized_pipe_points_2[:, 1], k=3)
            pipe_mask_rgb_2 = draw_pipe_mask(init_frame, self.x_smooth_2, self.y_smooth_2, pipe_width=pipe_width)
            self.pipe_mask_2 = cv2.cvtColor(pipe_mask_rgb_2, cv2.COLOR_BGR2GRAY)
            pipe_mask_rgb_yolo_2 = draw_pipe_mask(init_frame, self.x_smooth_2, self.y_smooth_2, pipe_width=pipe_width * 2)
            self.pipe_mask_yolo_2 = cv2.cvtColor(pipe_mask_rgb_yolo_2, cv2.COLOR_BGR2GRAY)
            self.pipe_mask = cv2.bitwise_or(self.pipe_mask, self.pipe_mask_2)
            self.pipe_mask_yolo = cv2.bitwise_or(self.pipe_mask_yolo, self.pipe_mask_yolo_2)

        self.milli_coords, self.milli_mask, milli_mask_image, have_milli = find_milli(
            init_frame, model_yolo, pipe_mask=self.pipe_mask_yolo
        )

        self.nano_coords, self.nano_mask = find_nano(
            init_frame, self.pipe_mask, self.milli_mask, threshold_nano=self.threshold_nano
        )

        self.circle_mask = create_circle_mask(init_frame, self.milli_coords, circle_diameter=0.1)

        self.intersection_mask, self.mean_gray = calculate_intersection_and_gray(
            init_frame, self.pipe_mask, self.circle_mask, self.milli_mask
        )

        return origin_frame, init_frame, self.milli_coords, self.nano_coords, self.mean_gray, self.x_smooth, self.y_smooth, self.spline, self.x_smooth_2, self.y_smooth_2, self.spline_2, self.pipe_mask, self.milli_mask, self.nano_mask, self.intersection_mask

    def get_info(self, choose_path_milli=0, choose_path_nano=0):
        """Get current frame and detect robot positions in their respective pipes."""
        origin_frame = self.get_image_init()
        frame = origin_frame.copy()

        if choose_path_milli == 0:
            pipe_mask_milli = self.pipe_mask_1
            pipe_mask_milli_yolo = self.pipe_mask_yolo_1
        else:
            pipe_mask_milli = self.pipe_mask_2 if self.pipe_mask_2 is not None else self.pipe_mask_1
            pipe_mask_milli_yolo = self.pipe_mask_yolo_2 if self.pipe_mask_yolo_2 is not None else self.pipe_mask_yolo_1

        if choose_path_nano == 0:
            pipe_mask_nano = self.pipe_mask_1
        else:
            pipe_mask_nano = self.pipe_mask_2 if self.pipe_mask_2 is not None else self.pipe_mask_1

        milli_coords, milli_mask, milli_mask_image, have_milli = find_milli(frame, model_yolo, pipe_mask=pipe_mask_milli_yolo)
        if milli_coords.size > 0:
            self.milli_coords = milli_coords

            if have_milli:
                self.milli_mask = milli_mask
                self.circle_mask = create_circle_mask(frame, self.milli_coords, circle_diameter=0.1)
                self.intersection_mask, self.mean_gray = calculate_intersection_and_gray(
                    frame, pipe_mask_milli, self.circle_mask, self.milli_mask
                )

            if not have_milli:
                self.mean_gray = 50

            print('mean_gray', self.mean_gray)

        self.nano_coords, self.nano_mask = find_nano(frame, pipe_mask_nano, self.milli_mask, threshold_nano=self.threshold_nano)

        return origin_frame, frame, self.milli_coords, self.nano_coords, self.mean_gray, self.milli_mask, self.nano_mask, self.intersection_mask


def find_nano(frame, pipe_mask, mask_millicore, threshold_nano):
    """Detect nanounits via intensity thresholding within the specified pipe, excluding millicore region."""
    pipe_inside_gray = cv2.bitwise_and(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY), pipe_mask)
    mask_millicore_not = cv2.bitwise_not(mask_millicore)
    pipe_gray = cv2.bitwise_and(pipe_inside_gray, mask_millicore_not)
    pipe_gray[pipe_gray == 0] = 255

    _, binary_mask = cv2.threshold(pipe_gray, threshold_nano, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

    num_labels, labels_im = cv2.connectedComponents(eroded_mask)
    output_frame = frame.copy()

    overlay = output_frame.copy()
    overlay[eroded_mask == 255] = (0, 255, 0)
    cv2.addWeighted(output_frame, 1, overlay, 0.5, 0)

    centerpoints = []
    areas = []

    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8) * 255
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            centerpoints.append((cx, cy))
            area = cv2.contourArea(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
            areas.append(area)

    area_center_pairs = sorted(zip(areas, centerpoints), key=lambda x: x[0], reverse=True)

    if area_center_pairs:
        sorted_areas, sorted_centerpoints = zip(*area_center_pairs)
        height, width = frame.shape[:2]
        nano_coords = np.array([(cx / width, cy / height) for cx, cy in sorted_centerpoints])

        unique_nano_coords = []
        if np.array(sorted_centerpoints).any():
            if nano_coords.ndim == 2:
                unique_nano_coords = nano_coords[0]
            elif nano_coords.ndim == 1:
                unique_nano_coords = nano_coords
    else:
        unique_nano_coords = np.array(-1)

    return unique_nano_coords, eroded_mask

def draw_pipe_mask(frame, x_smooth, y_smooth, pipe_width):
    frame_copy = frame.copy()
    mask_pipe = np.zeros_like(frame_copy, dtype=np.uint8)
    radius = int(pipe_width / 2 * min(frame_copy.shape[0], frame_copy.shape[1]))
    print('radius', radius)

    for x, y in zip(x_smooth, y_smooth):
        center = (int(x * frame_copy.shape[1]), int(y * frame_copy.shape[0]))
        cv2.circle(mask_pipe, center, radius, (255, 255, 255), thickness=-1)

    return mask_pipe


def create_circle_mask(frame, millicore_coord, circle_diameter):
    circle_mask = np.zeros_like(frame, dtype=np.uint8)
    radius = int(0.1 / 2 * min(frame.shape[0], frame.shape[1]))
    center = (int(millicore_coord[0] * frame.shape[1]), int(millicore_coord[1] * frame.shape[0]))
    cv2.circle(circle_mask, center, radius, (255, 255, 255), thickness=-1)
    return np.array(cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY))

def calculate_intersection_and_gray(frame, pipe_mask, circle_mask, mask_millicore):
    """Compute mean gray value near the millicore (excluding the millicore region)."""
    intersection_mask = cv2.bitwise_and(pipe_mask, circle_mask)
    mask_millicore_not = cv2.bitwise_not(mask_millicore)
    gray_mask = cv2.bitwise_and(intersection_mask, mask_millicore_not)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.bitwise_and(gray_frame, gray_mask)
    non_black_pixels = gray_roi[gray_mask > 0]

    if non_black_pixels.size == 0:
        print("No valid pixels found in the intersection region.")
        return intersection_mask, None

    sorted_pixels = np.sort(non_black_pixels)
    low_index = int(sorted_pixels.size * 0.3)
    high_index = int(sorted_pixels.size * 0.6)
    mean_gray = np.mean(sorted_pixels[low_index:high_index])

    return gray_mask, mean_gray

def resize_image(frame, scale_factor=0.25):
    frame_copy = frame.copy()
    new_width = int(frame_copy.shape[1] * scale_factor)
    new_height = int(frame_copy.shape[0] * scale_factor)
    frame_resized = cv2.resize(frame_copy, (new_width, new_height))
    return frame_resized

def find_milli(frame, model_yolo, pipe_mask=None):
    """Detect millicores using YOLO within the specified pipe region."""
    frame_copy = frame.copy()
    have_milli = False
    
    if pipe_mask is not None:
        if len(pipe_mask.shape) == 2:
            pipe_mask_3ch = cv2.cvtColor(pipe_mask, cv2.COLOR_GRAY2BGR)
        else:
            pipe_mask_3ch = pipe_mask.copy()
        mask_inv = cv2.bitwise_not(pipe_mask_3ch)
        frame_copy = cv2.bitwise_and(frame_copy, pipe_mask_3ch)
    
    results = model_yolo.predict(source=[frame_copy], conf=0.01, save=False, save_txt=False, verbose=False)
    
    print(f"Debug: results type={type(results)}, length={len(results)}")
    if len(results) > 0:
        print(f"Debug: results[0] type={type(results[0])}")
        if hasattr(results[0], 'boxes'):
            print(f"Debug: results[0].boxes={results[0].boxes}")
            if results[0].boxes is not None:
                print(f"Debug: results[0].boxes length={len(results[0].boxes)}")
    
    if len(results) == 0:
        print("Warning: YOLO did not return any results.")
        millicore_coords = np.array([-1, -1])
        millicore_mask = np.zeros_like(frame_copy[:, :, 0])
        return millicore_coords, millicore_mask, None, False
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        print("Warning: YOLO did not detect any objects.")
        millicore_coords = np.array([-1, -1])
        millicore_mask = np.zeros_like(frame_copy[:, :, 0])
        return millicore_coords, millicore_mask, None, False
    
    class_ids = result.boxes.cls.int().tolist()
    millicore_mask = np.zeros_like(frame_copy[:, :, 0])
    
    if hasattr(model_yolo, 'names'):
        print(f"\nDetected object classes:")
        unique_class_ids = list(set(class_ids))
        for class_id in unique_class_ids:
            class_name = model_yolo.names.get(int(class_id), f"Unknown class ({class_id})")
            count = class_ids.count(class_id)
            print(f"  ID {class_id}: {class_name} (detected {count})")
        print(f"All class IDs: {class_ids}")
    else:
        print(f"Detected class IDs: {class_ids}")
    
    if result.masks is None:
        print("Warning: YOLO result has no masks; model may not support segmentation or objects lack masks.")
        millicore_coords = []
        for i, class_id in enumerate(class_ids):
            class_name = model_yolo.names.get(int(class_id), f"Unknown({class_id})") if hasattr(model_yolo, 'names') else f"ID{class_id}"
            print(f"  Box {i}: class {class_name} (ID: {class_id})")
            if class_id == 1:
                box = result.boxes.xyxy[i].cpu().numpy()
                center_x = (box[0] + box[2]) / 2 / frame_copy.shape[1]
                center_y = (box[1] + box[3]) / 2 / frame_copy.shape[0]
                millicore_coords.append([center_x, center_y])
                have_milli = True
                print(f"    Using bounding box center: ({center_x:.3f}, {center_y:.3f})")
        
        if len(millicore_coords) == 0:
            unique_milli_coords = np.array([-1, -1])
        else:
            millicore_coords = np.array(millicore_coords)
            if millicore_coords.ndim == 2:
                unique_milli_coords = millicore_coords[0]
            elif millicore_coords.ndim == 1:
                unique_milli_coords = millicore_coords
            else:
                unique_milli_coords = np.array([-1, -1])
        
        return unique_milli_coords, millicore_mask, None, have_milli

    for i, class_id in enumerate(class_ids):
        class_name = model_yolo.names.get(int(class_id), f"Unknown({class_id})") if hasattr(model_yolo, 'names') else f"ID{class_id}"
        
        if result.masks is None or result.masks.data is None or i >= len(result.masks.data):
            print(f"Warning: detection {i} (class: {class_name}, ID: {class_id}) has no valid mask data")
            continue
            
        try:
            mask = (result.masks.data[i].cpu().numpy() * 255).astype("uint8")
            mask = cv2.resize(mask, frame_copy.shape[:2][::-1])
            if class_id == 1:
                millicore_mask = np.maximum(millicore_mask, mask)
                have_milli = True
                print(f"  Processed detection {i}: class {class_name} (ID: {class_id}) - added to millicore_mask")
        except Exception as e:
            print(f"Warning: failed to process mask for detection {i} (class: {class_name}, ID: {class_id}): {e}")
            continue

    millicore_coords = []
    if np.all(millicore_mask == 0):
        millicore_coords.append([-1, -1])
    else:
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(millicore_mask)
        for i in range(1, labels.max() + 1):
            x = centroids[i][0] / frame_copy.shape[1]
            y = centroids[i][1] / frame_copy.shape[0]
            millicore_coords.append([x, y])

    overlay = np.zeros_like(frame_copy)
    overlay[millicore_mask == 255] = (255, 0, 0)

    result_image = cv2.addWeighted(frame_copy, 1.0, overlay, 0.5, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for coord in millicore_coords:
        x, y = int(coord[0] * frame_copy.shape[1]), int(coord[1] * frame_copy.shape[0])
        cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(result_image, "Millicore", (x + 10, y), font, 1, (255, 0, 0), 2)

    unique_milli_coords = []
    millicore_coords = np.array(millicore_coords)
    if millicore_coords.size > 0:
        if millicore_coords.ndim == 2:
            unique_milli_coords = millicore_coords[0]
        elif millicore_coords.ndim == 1:
            unique_milli_coords = millicore_coords
        else:
            unique_milli_coords = np.array([-1, -1])
    else:
        unique_milli_coords = np.array([-1, -1])
    
    if isinstance(unique_milli_coords, list):
        unique_milli_coords = np.array(unique_milli_coords)
    if unique_milli_coords.ndim == 0 or (unique_milli_coords.ndim == 1 and len(unique_milli_coords) == 0):
        unique_milli_coords = np.array([-1, -1])
    elif unique_milli_coords.ndim > 1:
        unique_milli_coords = unique_milli_coords.flatten()[:2]

    return unique_milli_coords, np.array(millicore_mask), result_image, have_milli