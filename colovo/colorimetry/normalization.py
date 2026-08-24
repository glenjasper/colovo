#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def gray_world_normalization(image):
    img = image.astype(np.float32)

    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])

    avg = (avg_r + avg_g + avg_b) / 3

    img[:, :, 0] *= avg / avg_r
    img[:, :, 1] *= avg / avg_g
    img[:, :, 2] *= avg / avg_b

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)
