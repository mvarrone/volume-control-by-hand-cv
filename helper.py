from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import math
import numpy as np


def load_variables(name):
    if name == "camera":
        index_cam = 0  # Built-in webcam index
        width_cam = 1280
        height_cam = 720
        flipCode = 1  # 1 for horizontal flipping video
        window_name = "Running..."

        return index_cam, width_cam, height_cam, flipCode, window_name

    if name == "volume":
        volume = initialize_audio_settings()
        # volBar = 400
        # volPer = 0
        volPer = volume.GetMasterVolumeLevelScalar() * 100
        volPer = round(volPer, 2)
        volPer = math.ceil(volPer)
        volBar = np.interp(volPer, [0, 100], [400, 150])

        return volume, volPer, volBar

    if name == "fingers":
        area = 0
        fingers_dict = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20,
        }

        return area, fingers_dict


def initialize_audio_settings():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    # volRange = volume.GetVolumeRange()
    # minVol = volRange[0]
    # maxVol = volRange[1]

    return volume
