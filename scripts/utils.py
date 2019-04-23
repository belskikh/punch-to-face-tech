import numpy as np
import cv2

import re
import shutil
import time
import os
from pathlib import Path
from typing import Dict, Tuple, List, Union

import matplotlib
import matplotlib.pyplot as plt


IMG_EXT = ['.png', '.jpg', '.jpeg']


def create_overlays(
        root_dir: Union[str, Path],
        img_dir: Union[str, Path] = 'images',
        mask_dir: Union[str, Path] = 'masks',
        overlay_dir: Union[str, Path] = 'overlays',
        color: Tuple[int, int, int] = (89, 69, 15),
        alpha: float = 0.5,
        verbose: bool = True,
        ignore_errors: bool = False) -> None:
    # convert all paths to PosixPath
    root_dir = Path(root_dir)
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    overlay_dir = Path(overlay_dir)
    # if img_dir, mask_dir and overlay_dir
    # are not absolute paths
    if root_dir not in img_dir.parents:
        img_dir = root_dir / img_dir
    if root_dir not in mask_dir.parents:
        mask_dir = root_dir / mask_dir
    if root_dir not in overlay_dir.parents:
        overlay_dir = root_dir / overlay_dir
    # if directory for overlays doesn't exist
    overlay_dir.mkdir(parents=True, exist_ok=True)
    # iterate over image paths
    for fn in img_dir.iterdir():
        if str.lower(fn.suffix) not in IMG_EXT:
            continue
        img = cv2.imread(str(fn), cv2.IMREAD_COLOR)
        mask_fn = mask_dir / fn.name
        mask = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            if ignore_errors:
                continue
            else:
                raise FileNotFoundError(mask_fn)
        overlay = make_overlay(img, mask, color, alpha)
        ret = cv2.imwrite(str(overlay_dir / fn.name), overlay)
        if verbose:
            print(f'done with {fn.name}')


def make_overlay(
        img: np.ndarray, mask: np.ndarray,
        color: Tuple[int, int, int] = (89, 69, 15),
        alpha: float = 0.5) -> np.ndarray:
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


def sort_frame_func(filename: Path) -> int:
    matching_numbers = re.findall(r'\\d+', filename.stem)
    num_matches = len(matching_numbers)
    if num_matches == 0:
        return -1
    elif num_matches > 1:
        raise ValueError(f'Incorrect filename = {filename}')
    return int(matching_numbers[0])


def make_ordered_directory(directory: Union[str, Path]) -> None:
    res_dir = Path(directory)
    directory = Path(directory)
    millis = int(round(time.time() * 1000))
    tmp_dir = directory.parent / f'{millis}_{directory.name}'
    directory.replace(tmp_dir)
    res_dir.mkdir()
    files = sorted(tmp_dir.iterdir(), key=sort_frame_func)
    file_n = 0
    for fn in files:
        if str.lower(fn.suffix) not in IMG_EXT:
            continue
        new_fn = res_dir / f'{file_n}{fn.suffix}'
        shutil.move(str(fn), str(new_fn))
        file_n += 1
    shutil.rmtree(str(tmp_dir), ignore_errors=True)


def remove_redundant_channels(
        directory: Union[str, Path],
        verbose: bool = True) -> None:
    directory = Path(directory)
    files = directory.iterdir()
    for fn in files:
        if str.lower(fn.suffix) not in IMG_EXT:
            continue
        mask = cv2.imread(str(fn), -1)
        if len(mask.shape) < 3:
            continue
        mask = mask[:, :, 0]
        ret = cv2.imwrite(str(fn), mask)
        if verbose:
            print(f'done with {fn.name}')


def move_img_to_dir(
        src_dir: Union[str, Path],
        dst_dir: Union[str, Path],
        img_indices: List[int] = None) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fn in src_dir.iterdir():
        if fn.stem.isnumeric():
            flag = img_indices is None or int(fn.stem) in img_indices
            if flag:
                new_fn = dst_dir / fn.name
                shutil.move(str(fn), str(new_fn))


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def draw_points(
        img: np.ndarray,
        points: np.ndarray,
        mask: np.ndarray = None,
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 5,
        draw_ids: bool = False,
        point_ids: List[int] = None,
        font_scale: float = 1.0,
        font_thickness: int = 2) -> np.ndarray:

    draw_ids = draw_ids and point_ids is not None
    if draw_ids:
        flag = len(points) == len(point_ids)
        msg = 'There has to be equal number of points and point IDs'
        assert flag, msg
    # make copies
    # in order not to modify original objects
    img = img.copy()
    points = points.copy()
    # prepare points
    if points.shape[-1] > 2:
        points = points[..., 0:2]
    points = np.rint(points.reshape(-1, 2)).astype(np.int32)
    # apply mask if needed
    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)
    # iterate and draw
    for i, point in enumerate(points):
        point = tuple(point)
        cv2.circle(
            img=img,
            center=point,
            radius=radius,
            color=color,
            thickness=-1
        )
        # draw point IDs
        if draw_ids:
            text = str(point_ids[i])
            t_x = point[0] + radius * 2
            t_y = point[1] - radius * 2
            point = (t_x, t_y)
            cv2.putText(
                img=img,
                text=text,
                org=point,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=font_thickness
            )
    # return img with points drawn
    return img


def show_img(
        img: np.ndarray,
        mask: np.ndarray = None,
        fig_size: Tuple[float, float] = (10.0, 10.0)) -> None:

    new_fig_size = list(fig_size)
    orig_fig_size = list(matplotlib.rcParams['figure.figsize'])
    if new_fig_size != orig_fig_size:
        matplotlib.rcParams['figure.figsize'] = new_fig_size

    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)

    img_shape = img.shape

    if len(img_shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        if img_shape[2] == 4:
            img = img[..., 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

    plt.show()


def create_logo_texture(
        logo_fn: Union[str, Path],
        mask_fn: Union[str, Path],
        texture_size: Tuple[int, int],
        marker_size: Tuple[int, int],
        fraction: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:

    logo = cv2.imread(str(logo_fn), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)

    logo_w = int(marker_size[0] * fraction)
    logo_h = int(marker_size[1] * fraction)
    logo_size = (logo_w, logo_h)

    texture_shape = (texture_size[0], texture_size[1], 3)
    left = texture_shape[1] // 2 - logo_w // 2
    top = texture_shape[0] // 2 - logo_h // 2

    logo = cv2.resize(logo, logo_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, logo_size, interpolation=cv2.INTER_NEAREST)

    texture = np.zeros(shape=texture_shape, dtype=logo.dtype)
    texture_mask = np.zeros(shape=texture_size, dtype=mask.dtype)

    texture[top:top + logo_h, left:left + logo_w] = cv2.bitwise_and(
        logo, logo, mask=mask
    )
    texture_mask[top:top + logo_h, left:left + logo_w] = mask

    return texture, texture_mask


def load_frame_and_mask(
        frame_n: int,
        frame_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        frame_ext: str = '.jpg',
        mask_ext: str = '.png'):

    frame_path = frame_dir / f'{frame_n}{frame_ext}'
    mask_path = mask_dir / f'{frame_n}{mask_ext}'

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    return frame, mask
