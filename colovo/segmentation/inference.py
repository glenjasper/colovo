#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from .morphology import morphological_cleanup
from .postprocess import largest_component
from .models import build_segmentation_model
from configs.config import SEGMENTATION_THRESHOLD, IMAGE_SIZE

def load_segmentation_model(model_path):
    model = build_segmentation_model()
    model.load_state_dict(torch.load(model_path, map_location = "cpu"))
    model.eval()

    return model

def segment_yolk_ai(image, model):
    h, w = image.shape[:2]

    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0

    tensor = torch.tensor(img, dtype = torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor)[0][0].numpy()

    pred = cv2.resize(pred, (w, h), interpolation = cv2.INTER_LINEAR)
    mask = (pred > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255

    mask = morphological_cleanup(mask)
    mask = largest_component(mask)

    return mask
