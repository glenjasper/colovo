#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from configs.config import MORPH_KERNEL, MORPH_OPEN_ITERATIONS, MORPH_CLOSE_ITERATIONS

def morphological_cleanup(mask, kernel_size = MORPH_KERNEL):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = MORPH_OPEN_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = MORPH_CLOSE_ITERATIONS)

    return mask
