#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import cv2
from tqdm import tqdm
from collections import Counter
from colovo.segmentation.inference import load_segmentation_model
from configs.config import IMAGE_EXTENSIONS
from colovo.segmentation.pipeline import segment_with_fallback

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
OUTPUT_CSV = os.path.join(BASE_DIR, "models", "reports", "segmentation_report.csv")
OUTPUT_TXT = os.path.join(BASE_DIR, "models", "reports", "segmentation_report.txt")

SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolk_segmentation.pth")

def main():
    segmentation_model = load_segmentation_model(SEG_MODEL_PATH)

    files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    classical = 0
    unet = 0

    rows = []
    for filename in tqdm(files, desc = "Evaluando segmentação"):
        image = cv2.imread(os.path.join(IMAGE_DIR, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask, method, fallback_reason = segment_with_fallback(image, segmentation_model)
        mask_pixels = int((mask > 0).sum())

        rows.append({"image": filename,
                     "method": method,
                     "mask_pixels": mask_pixels,
                     "fallback_reason": fallback_reason})

        if method == "classical":
            classical += 1
        else:
            unet += 1

    fallback_reasons = Counter(row["fallback_reason"] for row in rows if row["fallback_reason"])

    total = len(files)

    classical_pct = 100 * classical / total
    unet_pct = 100 * unet / total

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok = True)

    with open(OUTPUT_CSV, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = ["image", "method", "mask_pixels", "fallback_reason"])
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT_TXT, "w", encoding = "utf-8") as f:
        f.write("COLOVO - Segmentation Report\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Total images......: {total}\n")
        f.write(f"Classical.........: {classical} ({classical_pct:.1f}%)\n")
        f.write(f"U-Net.............: {unet} ({unet_pct:.1f}%)\n")

        f.write("\nSegmentation reason(s)\n")
        f.write("------------------\n")
        if fallback_reasons:
            for reason, count in fallback_reasons.items():
                f.write(f"{reason}: {count}\n")
        else:
            f.write("None: 0\n")

    print()
    print("Segmentation report")
    print("===================")
    print(f"Total images......: {total}")
    print(f"Classical.........: {classical} ({classical_pct:.1f}%)")
    print(f"U-Net.............: {unet} ({unet_pct:.1f}%)")
    print()
    print(f"[INFO] CSV: {os.path.relpath(OUTPUT_CSV, BASE_DIR)}")
    print(f"[INFO] TXT: {os.path.relpath(OUTPUT_TXT, BASE_DIR)}")

if __name__ == "__main__":
    main()
