#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from configs.config import RF_ESTIMATORS, RF_RANDOM_STATE, DSM_MIN, DSM_MAX

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_FILE = os.path.join(BASE_DIR, "models", "dsm_training_features.csv")
OUTPUT_TXT = os.path.join(BASE_DIR, "models", "reports", "dsm_evaluation.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "models", "reports", "dsm_predictions.csv")

FEATURE_COLUMNS = ["hue_median",
                   "sat_median",
                   "val_median",
                   "lab_a_median",
                   "lab_b_median"]

def main():
    # =========================
    # Load extracted features
    # =========================

    df = pd.read_csv(FEATURES_FILE)

    required_columns = ["Photo", "DSM"] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError("Colunas ausentes no arquivo de características: " + ", ".join(missing_columns))

    # =========================
    # Prepare data
    # =========================

    X = df[FEATURE_COLUMNS].values
    y = df["DSM"].values

    loo = LeaveOneOut()

    y_true = []
    y_pred = []
    photos = []

    # =========================
    # Leave-One-Out
    # =========================

    print()
    print("Avaliação do modelo DSM")
    print("=======================")
    print(f"[INFO] Amostras: {len(df)}")
    print("[INFO] Método: Leave-One-Out Cross-Validation")
    print("[INFO] Modelo: Random Forest Regressor")
    print()

    for train_index, test_index in tqdm(loo.split(X), total = len(df), desc = "Evaluating", unit = "fold"):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        model = RandomForestRegressor(n_estimators = RF_ESTIMATORS, random_state = RF_RANDOM_STATE)
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)[0]

        # Restrict prediction to DSM scale
        prediction = max(DSM_MIN, min(DSM_MAX, prediction))

        true_value = float(y_test[0])
        photo = df.iloc[test_index[0]]["Photo"]

        y_true.append(true_value)
        y_pred.append(float(prediction))
        photos.append(photo)

        # print(f"[{iteration:04d}/{len(df):04d}] {photo} | DSM={true_value:g} | Predicted={prediction:.3f}")

    # =========================
    # Metrics
    # =========================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # =========================
    # Errors
    # =========================
    absolute_errors = np.abs(np.array(y_true) - np.array(y_pred))
    exact_accuracy = np.mean(np.round(y_pred) == np.array(y_true))
    within_1 = np.mean(absolute_errors <= 1)

    # =========================
    # Save predictions
    # =========================
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok = True)

    with open(OUTPUT_CSV, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Photo", "expected_DSM", "predicted_DSM", "absolute_error"])

        for i in range(len(y_true)):
            writer.writerow([photos[i], y_true[i], round(y_pred[i], 3), round(absolute_errors[i], 3)])

    # =========================
    # Save evaluation report
    # =========================
    with open(OUTPUT_TXT, "w", encoding = "utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("COLOVO - DSM Evaluation\n")
        f.write("=" * 85 + "\n\n")
        f.write("Validation method....: Leave-One-Out Cross-Validation\n")
        f.write("Model................: Random Forest Regressor\n")
        f.write(f"Samples..............: {len(df)}\n")
        f.write(f"Features.............: {', '.join(FEATURE_COLUMNS)}\n\n")
        f.write(f"MAE..................: {mae:.4f}\n")
        f.write(f"RMSE.................: {rmse:.4f}\n")
        f.write(f"R2...................: {r2:.4f}\n")
        f.write(f"Exact accuracy.......: {exact_accuracy:.4f}\n")
        f.write(f"Within ±1 DSM........: {within_1:.4f}\n")

    # =========================
    # Terminal output
    # =========================
    print()
    print("=======================")
    print("DSM Evaluation")
    print("=======================")
    print(f"Samples              : {len(df)}")
    print(f"MAE                  : {mae:.4f}")
    print(f"RMSE                 : {rmse:.4f}")
    print(f"R²                   : {r2:.4f}")
    print(f"Exact accuracy       : {exact_accuracy:.4f}")
    print(f"Within ±1 DSM        : {within_1:.4f}")
    print()
    print(f"[INFO] Relatório : {os.path.relpath(OUTPUT_TXT, BASE_DIR)}")
    print(f"[INFO] Predições : {os.path.relpath(OUTPUT_CSV, BASE_DIR)}")

if __name__ == "__main__":
    main()
