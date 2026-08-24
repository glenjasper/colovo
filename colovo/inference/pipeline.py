#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from colovo.segmentation.classical import segment_yolk_classical
from colovo.segmentation.inference import segment_yolk_ai
from colovo.colorimetry.features import extract_color_features, features_to_vector
from colovo.calibration.dsm_random_forest import predict_dsm
from colovo.segmentation.pipeline import segment_with_fallback

class ColovoPipeline:

    def __init__(self, segmentation_model, dsm_model):
        self.segmentation_model = segmentation_model
        self.dsm_model = dsm_model

    def predict(self, image):
        mask, method, fallback_reason = segment_with_fallback(image, self.segmentation_model)

        features = extract_color_features(image, mask)
        vector = features_to_vector(features)
        dsm = predict_dsm(self.dsm_model, vector)

        return {"mask": mask,
                "features": features,
                "dsm": dsm,
                "segmentation_method": method,
                "fallback_reason": fallback_reason}
