#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def load_image(file):
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
