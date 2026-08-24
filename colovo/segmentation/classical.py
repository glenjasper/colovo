#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from .morphology import morphological_cleanup
from .postprocess import best_component # largest_component
from configs.config import LOWER_HSV, UPPER_HSV

LOWER_HSV = np.array(LOWER_HSV, dtype = np.uint8)
UPPER_HSV = np.array(UPPER_HSV, dtype = np.uint8)

def threshold_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    return mask

def segment_yolk_classical(image):
    mask = threshold_hsv(image)
    mask = morphological_cleanup(mask)
    # mask = largest_component(mask)
    mask = best_component(mask, image)

    return mask
