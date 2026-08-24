#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from colovo.training.segmentation_trainer import SegmentationTrainer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "data", "segmentation", "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "data", "segmentation", "train", "masks")

VALIDATION_IMAGE_DIR = os.path.join(BASE_DIR, "data", "segmentation", "validation", "images")
VALIDATION_MASK_DIR = os.path.join(BASE_DIR, "data", "segmentation", "validation", "masks")

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolk_segmentation.pth")

def main():
    trainer = SegmentationTrainer(train_image_dir = TRAIN_IMAGE_DIR,
                                  train_mask_dir = TRAIN_MASK_DIR,
                                  validation_image_dir = VALIDATION_IMAGE_DIR,
                                  validation_mask_dir = VALIDATION_MASK_DIR,
                                  model_path = MODEL_PATH)
    trainer.train()

if __name__ == "__main__":
    main()
