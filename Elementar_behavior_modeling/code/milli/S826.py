"""Sensoray S826: analog outputs for magnetic coil drive (rotating field / setpoints).
Loads the vendor API DLL from ./devices/sdk_826_win_3.3.9/...;
"""
from ctypes import cdll
import ctypes
import os
import time
import math
import logging
import atexit

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DLL_PATH = os.path.join(
    _CODE_DIR, "devices", "sdk_826_win_3.3.9", "s826_3.3.9", "api", "x64", "s826.dll"
)
s826dll = cdll.LoadLibrary(_DLL_PATH)
BOARD = 0
# Per rangeCode: [lower_voltage_V, span_V] for S826_DacRangeWrite.
RANGE_PARAM = [[0, 5], [0, 10], [-5, 10], [-10, 20]]


class S826(object):
    """Wraps S826 DAC channels for triaxial field command and coil enable."""

    def __init__(self):
        self.lowerV = [-5, -5, -5, -5, -5, -5, -5, -5]
        self.rangeV = [10, 10, 10, 10, 10, 10, 10, 10]
        errcode = self.s826_init()
        if errcode != 1:
            print("Cannot detect S826 board. Error code: {}".format(errcode))
            self.s826_close()
            return
        self.s826_initRange()
        self.s826_setRange(chan=3, rangeCode=3)
        self.s826_setRange(chan=4, rangeCode=3)
        self.s826_setRange(chan=5, rangeCode=3)
        self.s826_setRange(chan=6, rangeCode=3)
        self.s826_setRange(chan=7, rangeCode=0)

    def s826_init(self):
        errCode = s826dll.S826_SystemOpen()
        return errCode

    def s826_close(self):
        s826dll.S826_SystemClose()

    def s826_initRange(self):
        for i in range(8):
            self.s826_setRange(i, 2)

    def s826_setRange(self, chan, rangeCode):
        """rangeCode: 0→[0,+5]V, 1→[0,+10]V, 2→[-5,+5]V, 3→[-10,+10]V."""
        self.lowerV[chan] = RANGE_PARAM[rangeCode][0]
        self.rangeV[chan] = RANGE_PARAM[rangeCode][1]
        s826dll.S826_DacRangeWrite(BOARD, chan, rangeCode, 0)

    def s826_aoPin(self, chan, outputB):
        """Single DAC channel; outputB in volts (sign per configured range)."""
        setpoint = int(32768 - 32768 / 10 * outputB)
        s826dll.S826_DacDataWrite(BOARD, chan, setpoint, 0)

    def setpoint(self, B_x, B_y, B_z):
        """Writes Bx (ch 3,6), By (4), Bz (5); ch 7 enables coil output."""
        setpoint_x = int(32768 + 3276.8 * 3 * B_x / 10)
        setpoint_y = int(32768 + 3276.8 * 3 * B_y / 10)
        setpoint_z = int(32768 + 3276.8 * 3 * B_z / 10)

        s826dll.S826_DacDataWrite(BOARD, 3, setpoint_x, 0) # X1
        s826dll.S826_DacDataWrite(BOARD, 6, setpoint_x, 0) # X2
        s826dll.S826_DacDataWrite(BOARD, 4, setpoint_y, 0) # Y
        s826dll.S826_DacDataWrite(BOARD, 5, setpoint_z, 0) # Z

        s826dll.S826_DacDataWrite(BOARD, 7, 43254, 0)

    def rotating_field(self, flux_density, frequency, pitch, direction):
        """Time-harmonic triaxial field; flux_density clamped to ±30 (implementation limit)."""
        A = flux_density
        if A > 30 or A < -30:
            print("Flux density out of allowed range.")
            return
        w = frequency
        p = pitch
        d = direction
        w1 = w * 2 * math.pi
        p1 = p * math.pi / 180
        d1 = d * math.pi / 180
        t = time.time()
        B_x = -A * math.cos(p1) * math.cos(d1) * math.cos(w1 * t) - A * math.sin(d1) * math.sin(w1 * t)
        B_y = -A * math.cos(p1) * math.sin(d1) * math.cos(w1 * t) + A * math.cos(d1) * math.sin(w1 * t)
        B_z = A * math.sin(p1) * math.cos(w1 * t)

        self.setpoint(B_x, B_y, B_z)

    def cleanup(self):
        self.rotating_field(flux_density=0, frequency=0, pitch=0, direction=0)
        print("Program exited, outputs set to 0.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

    coil = S826()
    coil.s826_setRange(chan=3, rangeCode=3)
    coil.s826_setRange(chan=6, rangeCode=3)
    coil.s826_setRange(chan=4, rangeCode=3)
    coil.s826_setRange(chan=5, rangeCode=3)
    coil.s826_setRange(chan=7, rangeCode=0)

    try:
        while True:
            coil.rotating_field(flux_density=10, frequency=10, pitch=0, direction=90)
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
    finally:
        coil.cleanup()
