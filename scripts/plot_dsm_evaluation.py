#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "models", "reports", "dsm_predictions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "reports")
PREDICTION_PLOT_PATH = os.path.join(OUTPUT_DIR, "dsm_prediction_plot.png")
ERROR_PLOT_PATH = os.path.join(OUTPUT_DIR, "dsm_error_plot.png")

def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Arquivo de predições não encontrado: {PREDICTIONS_PATH}")

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = ["Photo", "DSM", "Predicted_DSM"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(missing_columns))

    return df

def calculate_metrics(df):
    y_true = df["DSM"].values
    y_pred = df["Predicted_DSM"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    exact_accuracy = ((y_pred.round() == y_true).mean())
    within_one = ((abs(y_pred - y_true) <= 1).mean())
    mean_error = (y_pred - y_true).mean()

    return {"n": len(df),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "exact_accuracy": exact_accuracy,
            "within_one": within_one,
            "mean_error": mean_error}

def plot_predictions(df, metrics):
    plt.figure(figsize = (8, 8))
    plt.scatter(df["DSM"], df["Predicted_DSM"], alpha = 0.7)

    min_value = min(df["DSM"].min(), df["Predicted_DSM"].min())
    max_value = max(df["DSM"].max(), df["Predicted_DSM"].max())

    plt.plot([min_value, max_value], [min_value, max_value], linestyle = "--")
    plt.xlabel("DSM humano")
    plt.ylabel("DSM predito")
    plt.title("Random Forest - DSM humano vs. DSM predito")

    plt.xlim(min_value - 0.5, max_value + 0.5)
    plt.ylim(min_value - 0.5, max_value + 0.5)

    statistics = (f"N = {metrics['n']}\n"
                  f"MAE = {metrics['mae']:.4f}\n"
                  f"RMSE = {metrics['rmse']:.4f}\n"
                  f"R² = {metrics['r2']:.4f}\n"
                  f"Exact accuracy = {metrics['exact_accuracy']:.2%}\n"
                  f"Within ±1 DSM = {metrics['within_one']:.2%}")

    plt.text(0.04,
             0.96,
             statistics,
             transform = plt.gca().transAxes,
             verticalalignment = "top",
             bbox = dict(boxstyle = "round", alpha = 0.85))

    plt.grid(alpha = 0.3)
    plt.tight_layout()

    plt.savefig(PREDICTION_PLOT_PATH, dpi = 300)
    plt.close()

def plot_errors(df, metrics):
    if "Error" in df.columns:
        error = df["Error"]
    else:
        error = (df["Predicted_DSM"] - df["DSM"])

    plt.figure(figsize = (8, 5))
    plt.scatter(df["DSM"], error, alpha = 0.7)
    plt.axhline(0, linestyle = "--")

    plt.xlabel("DSM humano")
    plt.ylabel("Erro (DSM predito - DSM humano)")
    plt.title("Random Forest - Erro das predições")

    statistics = (f"N = {metrics['n']}\n"
                  f"MAE = {metrics['mae']:.4f}\n"
                  f"RMSE = {metrics['rmse']:.4f}\n"
                  f"Mean error = {metrics['mean_error']:.4f}")

    plt.text(0.04,
             0.96,
             statistics,
             transform = plt.gca().transAxes,
             verticalalignment = "top",
             bbox = dict(boxstyle = "round", alpha = 0.85))

    plt.grid(alpha = 0.3)
    plt.tight_layout()

    plt.savefig(ERROR_PLOT_PATH, dpi = 300)
    plt.close()

def main():
    print("[INFO] Carregando predições DSM...")

    df = load_predictions()

    print(f"[INFO] Predições carregadas: {len(df)}")

    metrics = calculate_metrics(df)
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    print()
    print("[INFO] Gerando gráfico DSM humano vs. predito")
    plot_predictions(df, metrics)
    print(f"[INFO] Gráfico salvo: {os.path.relpath(PREDICTION_PLOT_PATH, BASE_DIR)}")

    print()
    print("[INFO] Gerando gráfico de erros")
    plot_errors(df, metrics)
    print(f"[INFO] Gráfico salvo: {os.path.relpath(ERROR_PLOT_PATH, BASE_DIR)}")

if __name__ == "__main__":
    main()
