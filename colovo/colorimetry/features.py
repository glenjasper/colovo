#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from configs.config import INNER_EROSION_KERNEL, INNER_EROSION_ITERATIONS, MIN_SATURATION, MAX_VALUE, MIN_VALID_PIXELS

def extract_color_features(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    lab_a = lab[:, :, 1]
    lab_b = lab[:, :, 2]

    # Erosão interna para remover pixels de borda (efeito de transição/transparência)
    kernel = np.ones((INNER_EROSION_KERNEL, INNER_EROSION_KERNEL), np.uint8)
    inner_mask = cv2.erode(mask, kernel, iterations = INNER_EROSION_ITERATIONS)

    # Criação do filtro de pixels válidos baseado em limites de saturação e brilho
    valid = (
        (inner_mask == 255) &
        (sat > MIN_SATURATION) &
        (val < MAX_VALUE)
    )

    # Fallback seguro: se o filtro estrito eliminar muitos pixels, usa a máscara original limpa
    if np.sum(valid) < MIN_VALID_PIXELS:
        valid = (mask == 255)

    # Se mesmo assim não houver pixels, evita um travamento geral retornando um dicionário padrão ou erro controlado
    if np.sum(valid) == 0:
        raise ValueError("A máscara da gema não contém pixels válidos para extração de características.")

    # Extração estatística otimizada usando os pixels mascarados
    hue_pixels = hue[valid]
    sat_pixels = sat[valid]
    val_pixels = val[valid]
    a_pixels = lab_a[valid]
    b_pixels = lab_b[valid]

    features = {
        "hue_median": float(np.median(hue_pixels)),
        "sat_median": float(np.median(sat_pixels)),
        "val_median": float(np.median(val_pixels)),
        "lab_a_median": float(np.median(a_pixels)),
        "lab_b_median": float(np.median(b_pixels)),

        "hue_std": float(np.std(hue_pixels)),
        "sat_std": float(np.std(sat_pixels)),
        "pixels": int(np.sum(valid))
    }

    return features

def features_to_vector(features):
    return np.array([
        features["hue_median"],
        features["sat_median"],
        features["val_median"],
        features["lab_a_median"],
        features["lab_b_median"]
    ])
