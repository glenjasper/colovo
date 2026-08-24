#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from colovo.training.dsm_trainer import DSMTrainer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "dsm", "dsm_labels.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolk_segmentation.pth")
DSM_MODEL_PATH = os.path.join(BASE_DIR, "models", "dsm_random_forest.pkl")
FEATURES_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "dsm_training_features.csv")

def main():

    trainer = DSMTrainer(csv_path = CSV_PATH,
                         image_dir = IMAGE_DIR,
                         segmentation_model_path = SEGMENTATION_MODEL_PATH,
                         model_path = DSM_MODEL_PATH,
                         features_output_path = FEATURES_OUTPUT_PATH)
    trainer.train()

if __name__ == "__main__":
    main()
