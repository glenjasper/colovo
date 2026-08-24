#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from datetime import datetime
from configs.config import TRAIN_RATIO, RANDOM_SEED, IMAGE_EXTENSIONS

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_MASK_DIR = os.path.join(BASE_DIR, "data", "raw", "masks")

TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "data", "segmentation", "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "data", "segmentation", "train", "masks")

VALID_IMAGE_DIR = os.path.join(BASE_DIR, "data", "segmentation", "validation", "images")
VALID_MASK_DIR = os.path.join(BASE_DIR, "data", "segmentation", "validation", "masks")

SPLIT_FILE = os.path.join(BASE_DIR, "data", "segmentation", "dataset_split.txt")

def create_directories():
    directories = (TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VALID_IMAGE_DIR, VALID_MASK_DIR)
    for directory in directories:
        os.makedirs(directory, exist_ok = True)

def clear_directory(directory):
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        if os.path.isfile(path):
            os.remove(path)

def copy_pair(filename, image_dst, mask_dst):
    shutil.copy2(os.path.join(RAW_IMAGE_DIR, filename), os.path.join(image_dst, filename))
    shutil.copy2(os.path.join(RAW_MASK_DIR, filename), os.path.join(mask_dst, filename))

def save_split(train_files, validation_files):
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok = True)

    with open(SPLIT_FILE, "w", encoding = "utf-8") as f:
        f.write("# COLOVO dataset split\n")
        f.write(f"# Date: {datetime.now()}\n")
        f.write(f"# Train ratio: {TRAIN_RATIO:.0%}\n")
        f.write(f"# Random seed: {RANDOM_SEED}\n\n")

        f.write("[TRAIN]\n")

        for file in sorted(train_files):
            f.write(f"{file}\n")

        f.write("\n[VALIDATION]\n")

        for file in sorted(validation_files):
            f.write(f"{file}\n")

def main():
    create_directories()

    clear_directory(TRAIN_IMAGE_DIR)
    clear_directory(TRAIN_MASK_DIR)
    clear_directory(VALID_IMAGE_DIR)
    clear_directory(VALID_MASK_DIR)

    files = [
        f for f in os.listdir(RAW_IMAGE_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not files:
        raise RuntimeError("Nenhuma imagem encontrada.")

    for file in files:
        mask = os.path.join(RAW_MASK_DIR, file)

        if not os.path.exists(mask):
            raise RuntimeError(f"Máscara inexistente: {file}")

    random.seed(RANDOM_SEED)
    random.shuffle(files)

    split = int(len(files) * TRAIN_RATIO)

    train_files = files[:split]
    validation_files = files[split:]

    for file in train_files:
        copy_pair(file, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)

    for file in validation_files:
        copy_pair(file, VALID_IMAGE_DIR, VALID_MASK_DIR)

    save_split(train_files, validation_files)

    print(f"[INFO] Total............: {len(files)}")
    print(f"[INFO] Treinamento......: {len(train_files)} ({TRAIN_RATIO:.0%})")
    print(f"[INFO] Validação........: {len(validation_files)} ({1-TRAIN_RATIO:.0%})")
    print(f"[INFO] Split file.......: {os.path.relpath(SPLIT_FILE, BASE_DIR)}")

if __name__ == "__main__":
    main()
