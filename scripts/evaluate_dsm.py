#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from configs.config import RF_ESTIMATORS, RF_RANDOM_STATE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(BASE_DIR, "models", "dsm_training_features.csv")
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "data", "segmentation", "train", "images")
VALIDATION_IMAGE_DIR = os.path.join(BASE_DIR, "data", "segmentation", "validation", "images")
REPORT_DIR = os.path.join(BASE_DIR, "models", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "dsm_evaluation.txt")
PREDICTIONS_PATH = os.path.join(REPORT_DIR, "dsm_predictions.csv")

FEATURE_COLUMNS = ["hue_median",
                   "sat_median",
                   "val_median",
                   "lab_a_median",
                   "lab_b_median"]

class DSMEvaluator:

    def __init__(self):
        self.features_path = FEATURES_PATH
        self.train_image_dir = TRAIN_IMAGE_DIR
        self.validation_image_dir = VALIDATION_IMAGE_DIR
        self.report_path = REPORT_PATH
        self.predictions_path = PREDICTIONS_PATH

    def load_features(self):
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Arquivo de características não encontrado: {self.features_path}")

        df = pd.read_csv(self.features_path)

        required_columns = (["Photo", "DSM"] + FEATURE_COLUMNS)
        missing_columns = [column for column in required_columns if column not in df.columns]

        if missing_columns:
            raise ValueError("Colunas obrigatórias ausentes no CSV: " + ", ".join(missing_columns))

        return df

    def get_image_names(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Diretório não encontrado: {directory}")

        return {
            filename
            for filename in os.listdir(directory)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        }

    def split_dataset(self, df):
        train_images = self.get_image_names(self.train_image_dir)

        validation_images = self.get_image_names(self.validation_image_dir)

        train_df = df[df["Photo"].isin(train_images)].copy()
        validation_df = df[df["Photo"].isin(validation_images)].copy()

        if len(train_df) == 0:
            raise ValueError("Nenhuma imagem do CSV foi encontrada no conjunto de treinamento.")

        if len(validation_df) == 0:
            raise ValueError("Nenhuma imagem do CSV foi encontrada no conjunto de validação.")

        overlap = train_images.intersection(validation_images)

        if overlap:
            raise ValueError("Existem imagens presentes simultaneamente em treino e validação:\n" + "\n".join(sorted(overlap)))

        csv_images = set(df["Photo"])

        missing_from_split = (csv_images - train_images - validation_images)

        if missing_from_split:
            raise ValueError("Imagens do CSV não pertencem nem ao treino nem à validação:\n" + "\n".join(sorted(missing_from_split)))

        return train_df, validation_df

    def train_and_predict(self, train_df, validation_df):
        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df["DSM"].values

        X_validation = validation_df[FEATURE_COLUMNS].values
        y_validation = validation_df["DSM"].values

        model = RandomForestRegressor(n_estimators = RF_ESTIMATORS, random_state = RF_RANDOM_STATE)
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        return y_validation, predictions

    def calculate_metrics(self, y_true, predictions):
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)
        exact_accuracy = np.mean(np.round(predictions) == y_true)
        within_one = np.mean(np.abs(predictions - y_true) <= 1)

        return {"mae": mae,
                "rmse": rmse,
                "r2": r2,
                "exact_accuracy": exact_accuracy,
                "within_one": within_one}

    def save_predictions(self, validation_df, predictions):
        predictions_df = pd.DataFrame({"Photo": validation_df["Photo"].values,
                                       "DSM": validation_df["DSM"].values,
                                       "Predicted_DSM": predictions})

        predictions_df["Error"] = (predictions_df["Predicted_DSM"] - predictions_df["DSM"])
        predictions_df["Absolute_Error"] = (predictions_df["Error"].abs())

        os.makedirs(os.path.dirname(self.predictions_path), exist_ok = True)

        predictions_df.to_csv(self.predictions_path, index = False)

    def save_report(self, train_df, validation_df, metrics):
        os.makedirs(os.path.dirname(self.report_path), exist_ok = True)

        with open(self.report_path, "w", encoding = "utf-8") as f:
            f.write("COLOVO - DSM Evaluation\n")
            f.write("=" * 35 + "\n\n")

            f.write("Evaluation strategy\n")
            f.write("-------------------\n")
            f.write("Training / Validation split\n\n")
            f.write(f"Training samples.....: {len(train_df)}\n")
            f.write(f"Validation samples...: {len(validation_df)}\n\n")

            f.write("Random Forest\n")
            f.write("-------------\n")
            f.write(f"Estimators...........: {RF_ESTIMATORS}\n")
            f.write(f"Random state.........: {RF_RANDOM_STATE}\n\n")

            f.write("Metrics\n")
            f.write("-------\n")
            f.write(f"MAE..................: {metrics['mae']:.4f}\n")
            f.write(f"RMSE.................: {metrics['rmse']:.4f}\n")
            f.write(f"R²...................: {metrics['r2']:.4f}\n")
            f.write(f"Exact accuracy.......: {metrics['exact_accuracy']:.4f}\n")
            f.write(f"Within ±1 DSM........: {metrics['within_one']:.4f}\n")

    def evaluate(self):
        print(f"[INFO] Características: {self.features_path}")

        df = self.load_features()

        print(f"[INFO] Total de imagens: {len(df)}")

        train_df, validation_df = self.split_dataset(df)

        print(f"[INFO] Treinamento   : {len(train_df)} imagens")
        print(f"[INFO] Validação     : {len(validation_df)} imagens")
        print()
        print("[INFO] Treinando Random Forest com dados de treinamento...")

        y_validation, predictions = self.train_and_predict(train_df, validation_df)
        metrics = self.calculate_metrics(y_validation, predictions)

        self.save_predictions(validation_df, predictions)
        self.save_report(train_df, validation_df, metrics)

        print()
        print("=======================")
        print("DSM Evaluation")
        print("=======================")
        print(f"Training samples      : {len(train_df)}")
        print(f"Validation samples    : {len(validation_df)}")
        print(f"MAE                   : {metrics['mae']:.4f}")
        print(f"RMSE                  : {metrics['rmse']:.4f}")
        print(f"R²                    : {metrics['r2']:.4f}")
        print(f"Exact accuracy        : {metrics['exact_accuracy']:.4f}")
        print(f"Within ±1 DSM         : {metrics['within_one']:.4f}")
        print()
        print(f"[INFO] Evaluation : {os.path.relpath(self.report_path, BASE_DIR)}")
        print(f"[INFO] Predictions: {os.path.relpath(self.predictions_path, BASE_DIR)}")

def main():
    evaluator = DSMEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
