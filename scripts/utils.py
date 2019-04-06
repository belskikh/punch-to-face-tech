import numpy as np
import cv2
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
        ignore_exceptions: bool = False):
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
            if ignore_exceptions:
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
