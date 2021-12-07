import os
import random

from pathlib import Path
from urllib.request import urlretrieve, urlcleanup
from zipfile import ZipFile

import albumentations as albm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def show_test_image_quality(model, image, anime, savename=None, interactive=False, device='cpu'):
    imageT = torchvision.transforms.ToTensor()(Image.fromarray(image)).unsqueeze(0).to(device)
    output = np.uint8(model(imageT).squeeze(0).detach().permute(1, 2, 0).cpu().numpy() * 255.)
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 3),  # creates 1x3 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, [image, anime, output]):
        ax.imshow(im)
        
    if savename:
        p = Path(savename)
        if p.parents:
            p.parents[0].mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p))
    if interactive:
        plt.show()

def visualize_ds(model, dataset, nrows=4, savename=None, interactive=False, device='cpu'):
    model = model.to(device)
    imgs = []
    for i in range(nrows):
        i, l = dataset[i]
        img = np.uint8(i.permute(1, 2, 0).detach().cpu().numpy() * 255.)
        imgs.append(img)
        label = np.uint8(l.permute(1, 2, 0).detach().cpu().numpy() * 255.)
        imgs.append(label)
        t = i.unsqueeze(0).to(device)
        output = np.uint8(model(t).squeeze(0).detach().permute(1, 2, 0).cpu().numpy() * 255.)
        imgs.append(output)
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, 3),  # creates 1x3 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, imgs):
        ax.axis('off')
        ax.imshow(im)

    if savename:
        p = Path(savename)
        if p.parents:
            p.parents[0].mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p))

    if interactive:
        plt.show()


