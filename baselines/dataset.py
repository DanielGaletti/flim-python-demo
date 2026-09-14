"""
Dataset para treino supervisionado dos baselines.
Carrega pares (imagem original, GT mask) com augmentation.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class SODDataset(Dataset):
    """Dataset para Salient Object Detection supervisionado."""

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, orig_folder, label_folder, file_list,
                 orig_ext=".png", label_ext=".png",
                 size=256, augment=True):
        self.orig_folder  = orig_folder
        self.label_folder = label_folder
        self.orig_ext     = orig_ext
        self.label_ext    = label_ext
        self.size         = size
        self.augment      = augment

        if isinstance(file_list, str):
            with open(file_list) as f:
                names = [l.strip() for l in f if l.strip()]
        else:
            names = list(file_list)

        # Remove extensão se presente
        self.names = [os.path.splitext(n)[0] for n in names]

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        img_path = os.path.join(self.orig_folder,  name + self.orig_ext)
        lbl_path = os.path.join(self.label_folder, name + self.label_ext)
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        return img, lbl

    def _augment(self, img, lbl):
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)
        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            lbl = TF.vflip(lbl)
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
            lbl = TF.rotate(lbl, angle)
        # Random crop (80-100% of size)
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.0)
            w, h  = img.size
            nw, nh = int(w * scale), int(h * scale)
            i = random.randint(0, h - nh)
            j = random.randint(0, w - nw)
            img = TF.crop(img, i, j, nh, nw)
            lbl = TF.crop(lbl, i, j, nh, nw)
        # Color jitter (imagem apenas)
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
        return img, lbl

    def __getitem__(self, idx):
        img, lbl = self._load(self.names[idx])
        img = img.resize((self.size, self.size), Image.BILINEAR)
        lbl = lbl.resize((self.size, self.size), Image.NEAREST)

        if self.augment:
            img, lbl = self._augment(img, lbl)
            img = img.resize((self.size, self.size), Image.BILINEAR)
            lbl = lbl.resize((self.size, self.size), Image.NEAREST)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, self.MEAN, self.STD)

        lbl_np = np.array(lbl).astype(np.float32)
        lbl_np[lbl_np > 0] = 1.0
        lbl_t  = torch.from_numpy(lbl_np).unsqueeze(0)

        return img_t, lbl_t, self.names[idx]
