#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
from tqdm import tqdm
from colovo.colorimetry.features import extract_color_features
from colovo.calibration.dsm_random_forest import train_dsm_model
from colovo.segmentation.inference import load_segmentation_model
from colovo.utils.paths import display_path
from colovo.segmentation.pipeline import segment_with_fallback

class DSMTrainer:

    def __init__(self, csv_path, image_dir, segmentation_model_path, model_path, features_output_path):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.segmentation_model_path = segmentation_model_path
        self.model_path = model_path
        self.features_output_path = features_output_path

    def load_labels(self):
        df = pd.read_csv(self.csv_path)

        required_columns = ["Photo", "DSM"]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente no CSV: {column}")

        return df

    def extract_features(self, df):
        segmentation_model = load_segmentation_model(self.segmentation_model_path)

        features_rows = []
        total = len(df)

        print()
        print("Extração de características DSM")

        for _, row in tqdm(df.iterrows(), total = len(df), desc = "Extracting", unit = "img"):
            filename = str(row["Photo"])
            dsm = float(row["DSM"])

            file = os.path.join(self.image_dir, filename)

            if not os.path.exists(file):
                raise FileNotFoundError(f"Imagem não encontrada: {file}")

            image = cv2.imread(file)

            if image is None:
                raise RuntimeError(f"Não foi possível carregar: {file}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Classical HSV → U-Net fallback
            mask, method, fallback_reason = segment_with_fallback(image, segmentation_model)

            features = extract_color_features(image, mask)

            features_rows.append({"Photo": filename,
                                  "DSM": dsm,
                                  "hue_median": features["hue_median"],
                                  "sat_median": features["sat_median"],
                                  "val_median": features["val_median"],
                                  "lab_a_median": features["lab_a_median"],
                                  "lab_b_median": features["lab_b_median"],
                                  "hue_std": features["hue_std"],
                                  "sat_std": features["sat_std"],
                                  "pixels": features["pixels"],
                                  "segmentation_method": method,
                                  "fallback_reason": fallback_reason})

            # print(f"[{index + 1:04d}/{total:04d}] {filename} | DSM={dsm:g} | Segmentation={method}")

        classical_count = sum(row["segmentation_method"] == "classical" for row in features_rows)
        unet_count = sum(row["segmentation_method"] == "unet" for row in features_rows)

        print()
        print("[INFO] Segmentação utilizada:")
        print(f"       Classical : {classical_count}")
        print(f"       U-Net     : {unet_count}")

        return pd.DataFrame(features_rows)

    def save_features(self, features_df):
        os.makedirs(os.path.dirname(self.features_output_path), exist_ok = True)
        features_df.to_csv(self.features_output_path, index = False)

        print()
        print(f"[INFO] Características salvas em: {display_path(self.features_output_path, 'models')}")

    def train(self):
        labels_df = self.load_labels()

        print(f"[INFO] Imagens rotuladas: {len(labels_df)}")

        features_df = self.extract_features(labels_df)
        self.save_features(features_df)

        print()
        print("[INFO] Treinando o modelo DSM")
        train_dsm_model(features_df, self.model_path)

        print()
        print("[INFO] Treinamento DSM concluído.")
