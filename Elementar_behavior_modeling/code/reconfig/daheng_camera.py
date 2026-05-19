"""Daheng (Galaxy) USB3 camera via gxipy: RGB capture, ROI, downsample, optional white balance.

Requires a color camera; mono devices are closed after detection. First frame sets gray-world gains for `auto_white_balance`.
"""
import os
import sys

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEVICES_DIR = os.path.join(_CODE_DIR, "devices")
if _DEVICES_DIR not in sys.path:
    sys.path.insert(0, _DEVICES_DIR)

import gxipy as gx
from ctypes import *
from gxipy.gxidef import *
import numpy as np
from gxipy.ImageFormatConvert import *
import cv2


class CameraImageProcessor:
    """Opens device index 1, configures stream and ROI, returns downsampled RGB frames."""

    def __init__(self):
        self.device_manager = gx.DeviceManager()
        self.dev_num, self.dev_info_list = self.device_manager.update_all_device_list()
        if self.dev_num == 0:
            print("Number of enumerated devices is 0")
        self.cam = self.device_manager.open_device_by_index(1)
        self.camera_setting()
        self.first_image = self.get_camera_image()
        self.b, self.g, self.r = self.first_image.mean(axis=(0, 1))
        self.k = (self.b + self.g + self.r) / 3
        self.kb, self.kg, self.kr = self.k / self.b, self.k / self.g, self.k / self.r

    def get_best_valid_bits(self, pixel_format):
        valid_bits = DxValidBit.BIT0_7
        if pixel_format in (GxPixelFormatEntry.MONO8, GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8, GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8
                            , GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8, GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
            valid_bits = DxValidBit.BIT0_7
        elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.BAYER_GR10,
                              GxPixelFormatEntry.BAYER_RG10, GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10):
            valid_bits = DxValidBit.BIT2_9
        elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.BAYER_GR12,
                              GxPixelFormatEntry.BAYER_RG12, GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12):
            valid_bits = DxValidBit.BIT4_11
        elif pixel_format in (GxPixelFormatEntry.MONO14):
            valid_bits = DxValidBit.BIT6_13
        elif pixel_format in (GxPixelFormatEntry.MONO16):
            valid_bits = DxValidBit.BIT8_15
        return valid_bits

    def convert_to_RGB(self, raw_image):
        self.image_convert.set_dest_format(GxPixelFormatEntry.RGB8)
        valid_bits = self.get_best_valid_bits(raw_image.get_pixel_format())
        self.image_convert.set_valid_bits(valid_bits)

        buffer_out_size = self.image_convert.get_buffer_size_for_conversion(raw_image)
        output_image_array = (c_ubyte * buffer_out_size)()
        output_image = addressof(output_image_array)

        self.image_convert.convert(raw_image, output_image, buffer_out_size, False)
        if output_image is None:
            print("Failed to convert RawImage to RGBImage")
            return

        return output_image_array, buffer_out_size

    def camera_setting(self):
        remote_device_feature = self.cam.get_remote_device_feature_control()
        # Full-frame ROI (3880×1920).
        # if remote_device_feature.is_writable("OffsetX"):
        #     remote_device_feature.get_int_feature("OffsetX").set(320)
        # if remote_device_feature.is_writable("OffsetY"):
        #     remote_device_feature.get_int_feature("OffsetY").set(0)
        if remote_device_feature.is_writable("Width"):
            remote_device_feature.get_int_feature("Width").set(3880)
        if remote_device_feature.is_writable("Height"):
            remote_device_feature.get_int_feature("Height").set(1920)

        self.image_convert = self.device_manager.create_image_format_convert()

        self.image_process = self.device_manager.create_image_process()
        self.image_process_config = self.cam.create_image_process_config()
        self.image_process_config.enable_color_correction(False)

        pixel_format_value, pixel_format_str = remote_device_feature.get_enum_feature("PixelFormat").get()
        if Utility.is_gray(pixel_format_value):
            print("This sample does not support mono camera.")
            self.cam.close_device()

        trigger_mode_feature = remote_device_feature.get_enum_feature("TriggerMode")
        trigger_mode_feature.set("Off")

        if remote_device_feature.is_readable("GammaParam"):
            gamma_value = remote_device_feature.get_float_feature("GammaParam").get()
            self.image_process_config.set_gamma_param(gamma_value)
        else:
            self.image_process_config.set_gamma_param(1)
        if remote_device_feature.is_readable("ContrastParam"):
            contrast_value = remote_device_feature.get_int_feature("ContrastParam").get()
            self.image_process_config.set_contrast_param(contrast_value)
        else:
            self.image_process_config.set_contrast_param(0)

        self.cam.stream_on()

    def get_camera_image(self, downsample_factor=4):
        """Grabs one frame; returns empty array on failure. RGB8 path skips conversion."""
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed.")
            return np.array([])

        if raw_image.get_pixel_format() != GxPixelFormatEntry.RGB8:
            rgb_image_array, rgb_image_buffer_length = self.convert_to_RGB(raw_image)
            if rgb_image_array is None:
                return np.array([])
            numpy_image = np.frombuffer(rgb_image_array, dtype=np.ubyte, count=rgb_image_buffer_length). \
                reshape(raw_image.frame_data.height, raw_image.frame_data.width, 3)
        else:
            numpy_image = raw_image.get_numpy_array()

        if numpy_image is None:
            return np.array([])

        numpy_image = numpy_image[::downsample_factor, ::downsample_factor, :]

        return numpy_image

    def auto_white_balance(self, image):
        """Channel gains from constructor reference frame (gray-world)."""
        b, g, r = np.split(image, 3, axis=2)

        r = cv2.multiply(r, self.kr)
        g = cv2.multiply(g, self.kg)
        b = cv2.multiply(b, self.kb)
        return cv2.merge((r, g, b))

    def auto_dynamic_range(self, image, scale_factor, dynamic_range_factor):
        img_min = float(np.min(image))
        image = (image - img_min) * scale_factor * dynamic_range_factor
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)

    def fast_auto_enhance(self, image, dynamic_range_factor=1.0):
        """Global gain from mean intensity, then min–max normalize (optional strength via dynamic_range_factor)."""
        pixels = image.ravel().astype(np.float32)

        k = 1.0
        pixels_mean = np.mean(pixels)
        pixels_coeff = k / pixels_mean if pixels_mean != 0 else 1.0

        image = cv2.multiply(image, pixels_coeff)

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) * dynamic_range_factor
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def stream_off(self):
        self.cam.stream_off()

    def cleanup(self):
        self.cam.close_device()


if __name__ == "__main__":
    # Minimal live preview; white-balance gains are set in CameraImageProcessor.__init__.
    proc = CameraImageProcessor()
    try:
        while True:
            image = proc.get_camera_image()
            image = proc.auto_white_balance(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.size:
                cv2.imshow("Image", image)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        proc.stream_off()
        proc.cleanup()
