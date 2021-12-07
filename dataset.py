import os
from pathlib import Path
import cv2
import torch
from PIL import Image


class AnimeDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, train=True, subset=False, style='paprika', transforms=None, target_transforms=None):
        if train:
            self.root_dir = Path(root_dir) / ('subset' if subset else 'train')
        else:
            self.root_dir = Path(root_dir) / 'val'

        self.train = train
        self.subset = subset
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.images_dir = self.root_dir / 'faces'
        self.targets_dir = self.root_dir / style
        self.images = sorted(os.listdir(self.images_dir))
        self.targets = sorted(os.listdir(self.targets_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(cv2.imread(str(self.images_dir / self.images[idx]))[:, :, ::-1])
        target = cv2.imread(str(self.targets_dir / self.images[idx]))
        if target is None:
            print(str(self.targets_dir / self.images[idx]))
        target = Image.fromarray(target[:, :, ::-1])
        target = Image.fromarray(cv2.imread(str(self.targets_dir / self.images[idx]))[:, :, ::-1])
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            target = self.target_transforms(target)
        return image, target
