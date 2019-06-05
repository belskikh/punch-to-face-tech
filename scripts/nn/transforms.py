import albumentations as albu


def get_canvas_inference_transforms(
        min_height: int = 1080,
        min_width: int = 1920,
        divider: int = 32) -> albu.Compose:

    min_height = _check_and_get_new_side(min_height, divider)
    min_width = _check_and_get_new_side(min_width, divider)

    return albu.Compose([
        albu.PadIfNeeded(min_height=min_height, min_width=min_width, p=1.0),
        albu.Normalize(p=1.0)
    ], p=1)


def get_canvas_center_crop(height: int = 1080, width: int = 1920) -> albu.CenterCrop:
    return albu.CenterCrop(height=height, width=width, p=1.0)


# helper function
def _check_and_get_new_side(side: int, divider: int):
    remainder = side % divider
    if remainder != 0:
        side = ((side // divider) + 1) * divider
    return side
