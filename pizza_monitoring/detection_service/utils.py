# pizza_monitoring/detection_service/utils.py

import base64
import numpy as np
import cv2

def decode_base64_frame(b64_str):
    byte_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(byte_data, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
