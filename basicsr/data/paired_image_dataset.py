# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2
import numpy as np
import os
import torch
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

class ClassificationImageDataset(data.Dataset):
    def __init__(self, opt):
        super(ClassificationImageDataset, self).__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.phase = opt.get('phase','train')
        self.phase_folder = os.path.join(self.root_folder, self.phase)

        if not os.path.isdir(self.phase_folder):
            raise FileNotFoundError(
                f"Phase folder '{self.phase_folder}' does not exist. Check your dataroot and phase.")
        try:
            self.classes = sorted([d for d in os.listdir(self.phase_folder)
                                   if os.path.isdir(os.path.join(self.phase_folder, d))])
            if not self.classes:
                raise FileNotFoundError(f"No class subfolders found in '{self.phase_folder}'.")
        except Exception as e:
            print(f"Error reading class folders from {self.phase_folder}: {e}")
            raise

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        print(f"Found classes for phase '{self.phase}': {self.classes} -> {self.class_to_idx}")
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.phase_folder, class_name)
            try:
                image_names = sorted(
                    [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                    key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else x)
            except ValueError:
                print(f"Warning: Could not sort image names numerically in {class_dir}. Sorting alphabetically.")
                image_names = sorted(
                    [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for img_name in image_names:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))
        if not self.samples:
            raise RuntimeError(
                f"Found 0 images in subfolders of '{self.phase_folder}'. Supported image extensions: .png, .jpg, .jpeg")
        print(f"Found {len(self.samples)} images for phase '{self.phase}'.")

        if self.phase == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)
            self.train_size = opt.get('train_size', None)

    def __getitem__(self, index):
        index = index % len(self.samples)
        img_path, target = self.samples[index]
        target = torch.tensor(target, dtype=torch.long)
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        img_np = imfrombytes(img_bytes, flag='grayscale', float32=True)

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)
        elif img_np.ndim != 3:
            raise ValueError(f"Unexpected image dimensions after loading and expand_dims. Shape: {img_np.shape}")
        if self.phase == 'train':
            if self.geometric_augs:
                pass

        img_tensor_list = img2tensor(img_np, bgr2rgb=False, float32=True)
        img_tensor = img_tensor_list[0]
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 3:
            if img_tensor.shape[0] != 1:
                if img_tensor.shape[2] == 1:

                    img_tensor = img_tensor.permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected tensor shape from img2tensor for {img_path}: {img_tensor.shape}")
        else:
            raise ValueError(f"Unexpected tensor dimensions after img2tensor for {img_path}: {img_tensor.ndim}D")

        return {
            'img': img_tensor,
            'label': target,
            'img_path': img_path
        }

    def __len__(self):
        return len(self.samples)