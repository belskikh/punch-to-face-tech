import numpy as np
import torch
from albumentations import Compose, RGBShift, RandomGamma, RandomCrop, ShiftScaleRotate, VerticalFlip, HorizontalFlip, RandomRotate90, RandomBrightnessContrast, GaussNoise

def x_to_torch(x):
    return torch.from_numpy(np.moveaxis(x.astype(np.float32) / 255., -1, 0))

def y_to_torch(y):
    return torch.from_numpy(np.expand_dims(y.astype(np.float32) / 255., axis=0))

def open_fn(x):
    return {'features': x['image'], 'targets': x['mask']}

def targets_aug(transforms, targets):
    target = {}
    for i, mask in enumerate(targets[1:]):
        target['mask' + str(i)] = 'mask'
    return Compose(transforms, p=1, additional_targets=target)(image=targets[0],
                                                                mask=targets[1],
                                                                mask1=targets[2],
                                                                mask2=targets[3])
def transforms(targets):
    aug = targets_aug([RandomCrop(256, 256),
        ShiftScaleRotate(),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        RandomBrightnessContrast(),
        GaussNoise(), RGBShift(), RandomGamma()
    ], targets)
    return aug
