import numpy as np
import cv2

from pathlib import Path
from typing import Dict, Tuple, List, Union

import torch

from transforms import get_canvas_inference_transforms, get_canvas_center_crop
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

    # height and width from VIDEO INFO
    center_crop = get_canvas_center_crop(height=1080, width=1920)

    mask_threshold = 0.78

    for frame_n, image in data_loader:
        frame_n = frame_n.item()
        image = image.cuda()
        with torch.no_grad():
            outputs = torch.sigmoid(model(image))
            outputs = outputs > mask_threshold
            outputs = outputs.cpu().numpy()
        masks = [mask[0] for mask in outputs]
        masks = [center_crop(image=mask)['image'] for mask in masks]
        masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # NOT FINISHED YET
