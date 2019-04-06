import numpy as np
import cv2
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, List, Union


def create_overlays(
        root_dir: Union[str, Path],
        img_dir: Union[str, Path] = 'images',
        mask_dir: Union[str, Path] = 'masks',
        overlay_dir: Union[str, Path] = 'overlays',
        img_ext: str = '.png',
        color: Tuple[int, int, int] = (89, 69, 15),
        alpha: float = 0.5,
        verbose: bool = True,
        ignore_errors: bool = False):
    # convert all paths to PosixPath
    root_dir = Path(root_dir)
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    overlay_dir = Path(overlay_dir)
    # if img_dir, mask_dir and overlay_dir
    # are not absolute paths
    if root_dir not in img_dir.parents:
        img_dir = root_dir/img_dir
    if root_dir not in mask_dir.parents:
        mask_dir = root_dir/mask_dir
    if root_dir not in overlay_dir.parents:
        overlay_dir = root_dir/overlay_dir
    # if directory for overlays doesn't exist
    overlay_dir.mkdir(parents=True, exist_ok=True)
    # iterate over image paths
    for fn in img_dir.iterdir():
        if fn.suffix != img_ext:
            continue
        img = cv2.imread(str(fn), cv2.IMREAD_COLOR)
        mask_fn = mask_dir/fn.name
        mask = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            if ignore_errors:
                continue
            else:
                raise FileNotFoundError(mask_fn)
        overlay = make_overlay(img, mask, color, alpha)
        ret = cv2.imwrite(str(overlay_dir/fn.name), overlay)
        if verbose:
            print(f'done with {fn.name}')


def make_overlay(
        img: np.ndarray, mask: np.ndarray,
        color: Tuple[int, int, int] = (89, 69, 15),
        alpha: float = 0.5):
    # result img
    output = img.copy()
    # overlay mask
    overlay = np.zeros_like(img)
    overlay[:, :] = color
    # inverse mask
    mask_inv = cv2.bitwise_not(mask)
    # black-out the area of mask
    output = cv2.bitwise_and(output, output, mask=mask_inv)
    # take only region of mask from overlay mask
    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    # original img with opaque mask
    overlay = cv2.add(output, overlay)
    # original img with overlay mask (alpha opacity)
    output = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    return output


def sort_frame_func(filename: Path):
    matching_numbers = re.findall(r'\\d+', filename.stem)
    num_matches = len(matching_numbers)
    if num_matches == 0:
        return -1
    elif num_matches > 1:
        raise ValueError(f'Incorrect filename = {filename}')
    return int(matching_numbers[0])


def make_ordered_directory(directory: Union[str, Path]):
    res_dir = Path(directory)
    directory = Path(directory)
    millis = int(round(time.time() * 1000))
    tmp_dir = directory.parent/f'{millis}_{directory.name}'
    directory.replace(tmp_dir)
    res_dir.mkdir()
    files = sorted(tmp_dir.iterdir(), key=sort_frame_func)
    file_n = 0
    for fn in files:
        if str.lower(fn.suffix) not in ['.png', '.jpg', '.jpeg']:
            continue
        new_fn = res_dir/f'{file_n}{fn.suffix}'
        shutil.move(str(fn), str(new_fn))
        file_n += 1
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
