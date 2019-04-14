import numpy as np
import cv2

from pathlib import Path
from typing import Dict, Tuple, List, Union

import torch
from torch.nn import functional as F

from transforms import get_canvas_inference_transforms
from data import CanvasInferenceDataset, get_canvas_inference_dataloader
from models import AlbuNet


def get_model(weights_path: Union[str, Path]) -> torch.nn.Module:
    model = AlbuNet(pretrained=True).cuda()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def predict(
        frame_dir: Union[str, Path],
        output_dir: Union[str, Path],
        dataset: CanvasInferenceDataset,
        batch_size: int = 1):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height = 1080
    width = 1920
    divider = 32
    transforms = get_canvas_inference_transforms(
        min_height=height,
        min_width=width,
        divider=divider
    )

    dataset = CanvasInferenceDataset(
        frame_dir=frame_dir,
        transforms=transforms
    )
    data_loader = get_canvas_inference_dataloader(
        dataset=dataset,
        batch_size=batch_size
    )

    model = get_model(weights_path)
    # not finished
