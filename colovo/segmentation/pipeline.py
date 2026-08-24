#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from configs.config import MIN_MASK_PIXELS, FORCE_UNET
from colovo.segmentation.classical import segment_yolk_classical
from colovo.segmentation.inference import segment_yolk_ai

def segment_with_fallback(image, segmentation_model):
    """
    Segmenta a gema usando Classical HSV.
    Se a máscara for insuficiente, utiliza U-Net.
    """

    if FORCE_UNET:
        mask = segment_yolk_ai(image, segmentation_model)
        return mask, "unet", "forced"

    mask = segment_yolk_classical(image)

    if np.sum(mask > 0) >= MIN_MASK_PIXELS:
        return mask, "classical", ""

    mask = segment_yolk_ai(image, segmentation_model)

    return mask, "unet", "mask_too_small"
