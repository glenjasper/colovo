#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from configs.config import IMAGE_SIZE, IMAGE_EXTENSIONS

class YolkDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size = IMAGE_SIZE):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        image_files = {f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)}
        mask_files = {f for f in os.listdir(mask_dir) if f.lower().endswith(IMAGE_EXTENSIONS)}

        images_without_masks = image_files - mask_files
        masks_without_images = mask_files - image_files

        if images_without_masks:
            raise FileNotFoundError("Imagens sem máscara correspondente:\n" + "\n".join(sorted(images_without_masks)))

        if masks_without_images:
            raise FileNotFoundError("Máscaras sem imagem correspondente:\n" + "\n".join(sorted(masks_without_images)))

        self.files = sorted(image_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        image_path = os.path.join(self.image_dir, file)
        mask_path = os.path.join(self.mask_dir, file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis = 0)

        image = torch.tensor(image, dtype = torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype = torch.float32)

        return image, mask
