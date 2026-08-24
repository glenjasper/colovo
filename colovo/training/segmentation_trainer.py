#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from colovo.datasets.loaders import YolkDataset
from colovo.segmentation.models import build_segmentation_model
from configs.config import TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_LR, IMAGE_SIZE
from colovo.utils.paths import display_path

class SegmentationTrainer:

    def __init__(self, train_image_dir, train_mask_dir, validation_image_dir, validation_mask_dir, model_path):
        self.train_image_dir = train_image_dir
        self.train_mask_dir = train_mask_dir

        self.validation_image_dir = validation_image_dir
        self.validation_mask_dir = validation_mask_dir

        self.model_path = model_path

        self.batch_size = TRAIN_BATCH_SIZE
        self.epochs = TRAIN_EPOCHS
        self.lr = TRAIN_LR

    def save_training_summary(self, train_dataset, validation_dataset, best_validation_loss):
        summary_path = os.path.join(os.path.dirname(self.model_path), "reports", "training_segmentation_summary.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok = True)

        with open(summary_path, "w", encoding = "utf-8") as f:
            f.write("COLOVO - Segmentation Training Summary\n")
            f.write("=" * 45 + "\n\n")

            f.write(f"Date................: {datetime.now()}\n\n")

            f.write(f"Model...............: SimpleUNet\n")
            f.write(f"Image size..........: {IMAGE_SIZE}\n\n")

            f.write(f"Training images.....: {len(train_dataset)}\n")
            f.write(f"Validation images...: {len(validation_dataset)}\n\n")

            f.write(f"Epochs..............: {self.epochs}\n")
            f.write(f"Batch size..........: {self.batch_size}\n")
            f.write(f"Learning rate.......: {self.lr}\n\n")

            f.write(f"Best validation loss: {best_validation_loss:.6f}\n")

        return summary_path

    def save_training_history(self, history):
        history_path = os.path.join(os.path.dirname(self.model_path), "reports", "training_segmentation_history.csv")
        os.makedirs(os.path.dirname(history_path), exist_ok = True)

        with open(history_path, "w", newline = "", encoding = "utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "validation_loss"])

            for row in history:
                writer.writerow([row["epoch"], row["train_loss"], row["validation_loss"]])

        return history_path

    def train(self):
        train_dataset = YolkDataset(self.train_image_dir, self.train_mask_dir)
        validation_dataset = YolkDataset(self.validation_image_dir, self.validation_mask_dir)

        print(f"[INFO] Training images   : {len(train_dataset)}")
        print(f"[INFO] Validation images : {len(validation_dataset)}\n")

        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        validation_loader = DataLoader(validation_dataset, batch_size = self.batch_size, shuffle = False)

        model = build_segmentation_model()

        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)

        criterion = nn.BCELoss()

        best_validation_loss = float("inf")

        history = []
        best_epoch = 0
        for epoch in range(self.epochs):
            # ---------------------
            # TRAIN
            # ---------------------
            model.train()
            train_loss = 0

            for images, masks in train_loader:
                predictions = model(images)
                loss = criterion(predictions, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))

            # ---------------------
            # VALIDATION
            # ---------------------
            model.eval()
            validation_loss = 0

            with torch.no_grad():
                for images, masks in validation_loader:
                    predictions = model(images)
                    loss = criterion(predictions, masks)
                    validation_loss += loss.item()

            validation_loss /= max(1, len(validation_loader))

            print(f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {validation_loss:.4f}")

            history.append({"epoch": epoch + 1,
                            "train_loss": train_loss,
                            "validation_loss": validation_loss})

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                os.makedirs(os.path.dirname(self.model_path), exist_ok = True)
                torch.save(model.state_dict(), self.model_path)
                print(f"      Novo melhor modelo, modelo salvo.")
                best_epoch = epoch + 1

        summary_path = self.save_training_summary(train_dataset, validation_dataset, best_validation_loss)
        history_path = self.save_training_history(history)

        print()
        print("[INFO] Treinamento concluído.")
        print(f"[INFO] Melhor Val Loss: {best_validation_loss:.4f}")
        print(f"[INFO] Melhor Epoch   : {best_epoch}")
        print(f"[INFO] Modelo         : {display_path(self.model_path, 'models')}")
        print(f"[INFO] Histórico      : {display_path(history_path, 'models')}")
        print(f"[INFO] Resumo         : {display_path(summary_path, 'models')}")
