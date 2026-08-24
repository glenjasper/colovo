#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from configs.config import MIN_COMPONENT_AREA, MIN_COMPONENT_CIRCULARITY, MIN_COMPONENT_SOLIDITY, YOLK_HUE_MIN, YOLK_HUE_MAX, YOLK_MIN_SATURATION

def largest_component(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    largest = max(contours, key = cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    return clean_mask

def component_properties(contour, hsv):
    area = cv2.contourArea(contour)

    if area <= 0:
        return None

    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return None

    circularity = (4 * np.pi * area / (perimeter * perimeter))

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area <= 0:
        return None

    solidity = area / hull_area

    component_mask = np.zeros(hsv.shape[:2], dtype = np.uint8)

    cv2.drawContours(component_mask, [contour], -1, 255, -1)

    pixels = component_mask == 255
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]

    mean_hue = float(np.mean(hue[pixels]))
    mean_saturation = float(np.mean(saturation[pixels]))
    median_saturation = float(np.median(saturation[pixels]))
    p75_saturation = float(np.percentile(saturation[pixels], 75))

    return {"area": area,
            "circularity": circularity,
            "solidity": solidity,
            "mean_hue": mean_hue,
            "mean_saturation": mean_saturation,
            "median_saturation": median_saturation,
            "p75_saturation": p75_saturation}

def component_score(properties):
    if properties is None:
        return -1

    area = properties["area"]
    circularity = properties["circularity"]
    solidity = properties["solidity"]
    mean_hue = properties["mean_hue"]
    median_saturation = properties["median_saturation"]
    p75_saturation = properties["p75_saturation"]

    if area < MIN_COMPONENT_AREA:
        return -1

    if circularity < MIN_COMPONENT_CIRCULARITY:
        return -1

    if solidity < MIN_COMPONENT_SOLIDITY:
        return -1

    if median_saturation < YOLK_MIN_SATURATION:
        return -1

    if not (YOLK_HUE_MIN <= mean_hue <= YOLK_HUE_MAX):
        return -1

    saturation_score = p75_saturation / 255.0
    score = (circularity * 0.30 + solidity * 0.20 + saturation_score * 0.50)

    return score

def best_component(mask, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    best = None
    best_score = -1

    for contour in contours:
        properties = component_properties(contour, hsv)
        score = component_score(properties)

        if score > best_score:
            best_score = score
            best = contour

    clean = np.zeros_like(mask)

    if best is not None:
        cv2.drawContours(clean, [best], -1, 255, -1)

    return clean
