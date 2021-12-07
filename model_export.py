from pathlib import Path
from urllib.request import urlretrieve, urlcleanup
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

class ModelForExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x) * 255


def export_model(model, modelname, height, width):
    model.eval().cpu()
    model_for_export = ModelForExport(model)
    dummy_input = torch.randn(1, 3, height, width,
                          dtype=torch.float32)

    output = model_for_export(dummy_input.detach())

    input_names = ["data"]
    output_names = ["out"]

    p = Path(modelname)
    if p.parents:
        p.parents[0].mkdir(parents=True, exist_ok=True)

    torch.onnx.export(model_for_export, dummy_input, 
                      modelname, verbose=False, 
                    input_names=input_names, output_names=output_names)

