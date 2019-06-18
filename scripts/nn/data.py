import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from typing import Dict, Tuple, List, Union

import albumentations as albu

# dirty hack
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from utils import IMG_EXT, num_cpus
from video_utils import VideoScene


class CanvasInferenceDataset(Dataset):

    def __init__(
            self,
            frame_dir: Union[str, Path],
            transforms: albu.Compose,
            video_scene: VideoScene = None) -> None:

        frame_dir = Path(frame_dir)
        files = [fn for fn in frame_dir.iterdir() if _check_frame_path(
            fn, video_scene
        )]
        files = sorted(files, key=_frame_sort_func)

        self.files = files
        self.frame_dir = frame_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray]:
        image_file_path = self.files[idx]
        image = _load_image(image_file_path)
        frame_num = int(image_file_path.stem)

        augmented = self.transforms(image=image)

        image = augmented['image']
        image = image.transpose((2, 0, 1))

        return frame_num, image


def get_canvas_inference_dataloader(
        dataset: CanvasInferenceDataset,
        batch_size: int = 1,
        num_workers: int = None) -> DataLoader:

    if num_workers is None:
        num_workers = num_cpus() // 2

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return data_loader


def _load_image(path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: Union[str, Path]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask


def _check_frame_path(fn: Path, video_scene: VideoScene = None) -> bool:
    flag = str.lower(fn.suffix) in IMG_EXT

    frame_n = fn.stem
    flag = flag and frame_n.isnumeric()

    if video_scene is None or not flag:
        return flag
    frame_n = int(frame_n)
    return video_scene.contains(frame_n)


def _frame_sort_func(fn: Path) -> int:
    return int(fn.stem)
