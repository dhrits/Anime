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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import argparse

from dataset import *
from loss import *
from metrics import *
from student import *
from visualization import *
from model_export import *


def random_seed(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # Flip values for slower training speed, but more determenistic results.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="Training data batch size", type=int, default=42)
    parser.add_argument("-w", "--workers", help="Number of workers for dataloading", type=int, default=2)
    parser.add_argument("-s", "--steps", help="Number of steps through training data", type=int, default=200000)
    parser.add_argument("--validation_frequency", help="Frequency with which to provide validation stats", type=int, default=200)
    parser.add_argument("-iw", "--input_width", help="Resize width of input for training", type=int, default=256)
    parser.add_argument("-ih", "--input_height", help="Resize height of input for training", type=int, default=256)
    parser.add_argument("-d", "--device", help="Training device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--onnx_export_name", help="Export name for onnx model", type=str, default="model.onnx")
    parser.add_argument("--model_progress_path", help="Path where to store pytorch model as it trains", type=str,
               default="model_progress.pth")
    parser.add_argument("--test_img", help="Path to test image", type=str, default="test.png")
    parser.add_argument("--test_anime", help="Path to test anime", type=str, default="test_paprika.png")
    parser.add_argument("--tb_path", help="Path to tensorboard experiments folder", type=str, default="runs/anime_experiment")
    parser.add_argument("--random_seed", help="Random seed", type=int, default=1337)
    parser.add_argument("--data_root", help="Root of dataset", type=str, default="../data/CelebA/")
    parser.add_argument("--anime_style", help="Style of anime", type=str, default="paprika",
               choices=["paprika", "webtoon", "face_v2"])
    parser.add_argument("--no_perceptual", help="Train without perceptual loss", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    BATCH_SIZE = args.batch_size  # number of images per batch
    NUM_WORKERS = args.workers  # number of CPU threads available for image preprocessing
    NUM_TRAINING_STEPS = args.steps
    if NUM_TRAINING_STEPS < 6000:
        NUM_TRAINING_STEPS = 6000
    VALIDATION_FREQUENCY = args.validation_frequency  # log validation every N steps

    INPUT_WIDTH = args.input_width
    INPUT_HEIGHT = args.input_height
    DEVICE = torch.device(args.device)
    if not torch.cuda.is_available() and args.device == "cuda":
        print("Cuda not available. Defaulting to CPU")
        DEVICE = torch.device('cpu')
    RANDOM_SEED = args.random_seed
    EXPORT_NAME = args.onnx_export_name
    MODEL_PROGRESS_PATH = args.model_progress_path
    TEST_IMG = cv2.imread(args.test_img)[:, :, ::-1]
    TEST_ANIME = cv2.imread(args.test_anime)[:, :, ::-1]
    TB_PATH = args.tb_path
    DATA_ROOT = args.data_root
    STYLE = args.anime_style

    writer = SummaryWriter(TB_PATH)

    random_seed(RANDOM_SEED)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((INPUT_WIDTH, INPUT_HEIGHT)),
        torchvision.transforms.ToTensor(),
    ])

    target_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((INPUT_WIDTH, INPUT_HEIGHT)),
        torchvision.transforms.ToTensor(),
    ])

    train_ds = AnimeDataset(DATA_ROOT, train=True, subset=False, style=STYLE, transforms=train_transforms,
                            target_transforms=target_transforms)
    val_ds = AnimeDataset(DATA_ROOT, train=False, style=STYLE, transforms=train_transforms,
                          target_transforms=target_transforms)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                               num_workers=NUM_WORKERS, drop_last=True,
                                               shuffle=True,
                                               worker_init_fn=lambda _: np.random.seed())
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKERS, drop_last=False,
                                             shuffle=False,
                                             worker_init_fn=lambda _: np.random.seed())


    model = AnimeNet()
    model = model.to(device=DEVICE)
    loss = AnimeLoss(perceptual=(not args.no_perceptual))
    loss = loss.to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       factor=.5, patience=3,
                                                       verbose=True)

    metrics = Metrics(loss)

    progress = tqdm(desc='Training progress', total=NUM_TRAINING_STEPS,
                dynamic_ncols=True, leave=False)

    batch_num = 0
    best_loss = float("inf")
    progress_counter = 0
    while batch_num < NUM_TRAINING_STEPS:
        train_loss = 0
        model.train()
        bn = 0
        for batch in train_loader:
            batch_num += 1
            optimizer.zero_grad()
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            loss_value = loss(pred, y)
            loss_value.backward()
            optimizer.step()

            train_loss = (train_loss * bn + loss_value.item()) / (bn + 1)
            writer.add_scalar('Loss/train', train_loss, bn)
            bn += 1

            # Next two lines make sure we don't update the progress bar too frequently
            if batch_num % int(NUM_TRAINING_STEPS / 1000) == 0:
                progress.update(int(NUM_TRAINING_STEPS / 1000))

            if batch_num % VALIDATION_FREQUENCY == 0:
                print("[{}] train \t loss: {:.5f}".format(batch_num, train_loss))
                train_loss = 0
                bn = 0

                model.eval()
                for batch_val in val_loader:
                    x, y = batch_val
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    prediction = model(x).detach()
                    metrics.add_batch(prediction, y)

                print("[{}] val \t loss: {:.5f}\t ".format(
                    batch_num, metrics.loss_value))
                scheduler.step(metrics.loss_value)
                writer.add_scalar('Loss/val', metrics.loss_value, batch_num)
                if metrics.loss_value < best_loss:
                    print("Loss improved. Saving model")
                    best_loss = metrics.loss_value
                    torch.save(model.state_dict(), MODEL_PROGRESS_PATH)
                show_test_image_quality(model, TEST_IMG, TEST_ANIME,
                                        savename='progress/' + str(progress_counter + 1) + '.png', device=DEVICE)
                progress_counter += 1
                metrics.reset_metrics()
                model.train()

            if batch_num >= NUM_TRAINING_STEPS:
                break
    progress.close()

    export_model(model, EXPORT_NAME, INPUT_HEIGHT, INPUT_WIDTH)
