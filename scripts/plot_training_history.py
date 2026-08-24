#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from colovo.utils.paths import display_path

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "models", "reports", "training_segmentation_history.csv")
PNG_OUTPUT = os.path.join(BASE_DIR, "models", "reports", "training_segmentation_history.png")
PDF_OUTPUT = os.path.join(BASE_DIR, "models", "reports", "training_segmentation_history.pdf")

def main():
    df = pd.read_csv(CSV_PATH)

    best_index = df["validation_loss"].idxmin()
    best_epoch = int(df.loc[best_index, "epoch"])
    best_validation_loss = (df.loc[best_index, "validation_loss"])
    final_train_loss = (df["train_loss"].iloc[-1])
    final_validation_loss = (df["validation_loss"].iloc[-1])
    final_gap = (final_validation_loss - final_train_loss)
    total_epochs = len(df)

    plt.figure(figsize = (8, 6))
    plt.plot(df["epoch"], df["train_loss"], label = "Training", linewidth = 2)
    plt.plot(df["epoch"], df["validation_loss"], label = "Validation", linewidth = 2)

    plt.scatter(best_epoch, best_validation_loss, s = 60, zorder = 5)
    # plt.annotate(f"Best: {best_validation_loss:.4f}", (best_epoch, best_validation_loss), xytext = (8, 8), textcoords = "offset points")

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title("Segmentation Training History")

    # plt.xticks(df["epoch"])

    statistics = (f"Epochs = {total_epochs}\n"
                  f"Best Validation Loss = {best_validation_loss:.4f}\n"
                  f"Best Epoch = {best_epoch}\n"
                  f"Final Train Loss = {final_train_loss:.4f}\n"
                  f"Final Validation Loss = {final_validation_loss:.4f}\n"
                  f"Final Gap = {final_gap:.4f}")

    plt.text(0.97,
             0.97,
             statistics,
             transform = plt.gca().transAxes,
             verticalalignment = "top",
             horizontalalignment = "right",
             bbox = dict(boxstyle = "round", alpha = 0.85))

    plt.grid(True, alpha = 0.25, linewidth = 0.6)
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.savefig(PNG_OUTPUT, dpi = 300)
    plt.savefig(PDF_OUTPUT)
    plt.close()

    print(f"[INFO] PNG Figure saved: {display_path(PNG_OUTPUT, 'models')}")
    print(f"[INFO] PDF Figure saved: {display_path(PDF_OUTPUT, 'models')}")

if __name__ == "__main__":
    main()
