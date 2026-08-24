#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from tqdm import tqdm
from colovo.segmentation.classical import segment_yolk_classical
from configs.config import IMAGE_EXTENSIONS

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
MASK_DIR = os.path.join(BASE_DIR, "data", "raw", "masks")

def main():
    os.makedirs(MASK_DIR, exist_ok = True)

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)]

    if not files:
        raise RuntimeError("Nenhuma imagem encontrada.")

    for file in tqdm(files, desc = "Gerando máscaras"):
        image_path = os.path.join(IMAGE_DIR, file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = segment_yolk_classical(image)
        output_path = os.path.join(MASK_DIR, file)

        cv2.imwrite(output_path, mask)

    print("[INFO] Máscaras geradas.")

if __name__ == "__main__":
    main()
