import numpy as np
import cv2

from pathlib import Path
from typing import Dict, Tuple, List, Union

from tqdm import tqdm

import torch

import albumentations as albu

# dirty hack
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from utils import make_overlay

from nn.transforms import get_canvas_inference_transforms, get_canvas_center_crop
from nn.data import CanvasInferenceDataset, get_canvas_inference_dataloader
from nn.models import AlbuNet


# height and width should be from VideoInfo
def predict(
        frame_dir: Union[str, Path],
        output_dir: Union[str, Path],
        weights_path: Union[str, Path],
        batch_size: int = 1,
        mask_threshold: float = 0.78,
        height: int = 1080,
        width: int = 1920,
        divider: int = 32,
        extension: str = '.jpg',
        save_overlays: bool = True,
        overlay_color: Tuple[int, int, int] = (89, 69, 15)) -> None:

    # create output directories
    output_dir = Path(output_dir)
    output_frame_dir = output_dir / 'frames'
    output_mask_dir = output_dir / 'masks'
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    if save_overlays:
        output_overlay_dir = output_dir / 'overlays'
        output_overlay_dir.mkdir(parents=True, exist_ok=True)

    # inference transformations
    transforms = get_canvas_inference_transforms(
        min_height=height,
        min_width=width,
        divider=divider
    )

    # dataloader
    dataset = CanvasInferenceDataset(
        frame_dir=frame_dir,
        transforms=transforms
    )
    data_loader = get_canvas_inference_dataloader(
        dataset=dataset,
        batch_size=batch_size
    )

    # AlbuNet
    model = _get_model(weights_path)

    # center crop mask
    # if it was padded
    mask_crop = get_canvas_center_crop(height=height, width=width)

    for frames, image in tqdm(data_loader):
        frames = list(frames.numpy())
        image = image.cuda()
        with torch.no_grad():
            outputs = torch.sigmoid(model(image))
            outputs = outputs > mask_threshold
            outputs = outputs.cpu().numpy()

        masks = _get_masks(outputs, mask_crop)
        _save_images(masks, output_mask_dir, frames, extension='.png')

        images = _load_images(frame_dir, frames, extension)
        if save_overlays:
            overlays = _create_overlays(images, masks, overlay_color)
            _save_images(overlays, output_overlay_dir, frames)

        images = _apply_masks(images, masks)
        _save_images(images, output_frame_dir, frames)


def _get_model(weights_path: Union[str, Path]) -> torch.nn.Module:
    model = AlbuNet(pretrained=True).cuda()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def _get_masks(
        outputs: np.ndarray,
        crop_fn: albu.CenterCrop) -> List[np.ndarray]:
    masks = [mask[0] for mask in outputs]
    masks = [crop_fn(image=mask)['image'] for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]
    return masks


def _save_images(
        images: List[np.ndarray],
        output_dir: Union[str, Path],
        frames: List[int],
        extension: str = '.jpg') -> None:

    output_dir = Path(output_dir)

    for idx, frame_n in enumerate(frames):
        filename = output_dir / f'{frame_n}{extension}'
        ret = cv2.imwrite(str(filename), images[idx])


def _apply_masks(
        images: List[np.ndarray],
        masks: List[np.ndarray]) -> List[np.ndarray]:

    return [cv2.bitwise_and(images[i], images[i], mask=masks[i])
            for i in range(len(images))]


def _create_overlays(
        images: List[np.ndarray],
        masks: List[np.ndarray],
        color: Tuple[int, int, int] = (89, 69, 15)) -> List[np.ndarray]:

    return [make_overlay(images[i], masks[i], color)
            for i in range(len(images))]


def _load_images(
        frame_dir: Union[str, Path],
        frames: List[int],
        extension: str = '.jpg') -> List[np.ndarray]:
    frame_dir = Path(frame_dir)
    path_templ = str(frame_dir / f'{{0}}{extension}')
    images = [_load_image(path_templ.format(frame_n)) for frame_n in frames]
    return images


def _load_image(path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)
