import os
import collections
import numpy as np
import cv2
from skimage.io import imsave
from data import Data
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import albumentations as albu
from catalyst.data.augmentor import Augmentor, AugmentorKeys
from catalyst.contrib.models import UNet
from catalyst.dl.utils import UtilsFactory
from catalyst.dl.runner import SupervisedModelRunner
from catalyst.dl.callbacks import (InferCallback, CheckpointCallback)

def open_fn(x):
    return {'features': x['image'], 'targets': x['mask']}

def aug_x_to_torch(x):
    return torch.from_numpy(np.moveaxis(x.astype(np.float32) / 255., -1, 0))

def aug_y_to_torch(y):
    return  torch.from_numpy(np.expand_dims(y.astype(np.float32) / 255., axis= 0))

class Infer:
    def __init__(self, data_dir, logs_dir, rez_dir, batch_size):
        super(Infer, self).__init__()
        self.__data_dir = data_dir
        self.__logs_dir = logs_dir
        self.__rez_dir = rez_dir
        self.__batch_size = batch_size
        self.__data = None
        self.__sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def __get_data(self):
        images_dir = os.path.join(self.__data_dir, 'frames')
        images_paths = [f for f in os.listdir(images_dir) if f.split('.')[1] == 'jpg']

        data = Data(images_paths, self.__data_dir)
        self.__data = data
        data_transforms = transforms.Compose([
            AugmentorKeys(
                dict2fn_dict={'features': 'image', 'targets': 'mask'},
                augment_fn=albu.Resize(256, 256)
            ),
            Augmentor(
                dict_key='features',
                augment_fn=aug_x_to_torch
            ),
            Augmentor(
                dict_key='features',
                augment_fn=transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            ),
            Augmentor(
                dict_key='targets',
                augment_fn=aug_y_to_torch
            )
        ])

        loaders = collections.OrderedDict()

        test_loader = UtilsFactory.create_loader(
            data_source= data,
            open_fn= open_fn,
            dict_transform= data_transforms,
            batch_size= self.__batch_size,
            workers= 2,
            shuffle= False
        )

        loaders['infer'] = test_loader

        return loaders

    def __set_callbacks(self):
        callbacks = collections.OrderedDict()

        callbacks['saver'] = CheckpointCallback(
            resume=f'{self.__logs_dir}/checkpoint.best.pth.tar')
        callbacks['infer'] = InferCallback()

        return callbacks

    def inference(self):
        model = UNet(num_classes=1, in_channels= 3, num_filters= 64, num_blocks= 4)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        callbacks = self.__set_callbacks()
        loaders = self.__get_data()
        runner = SupervisedModelRunner(
            model=model,
            criterion=criterion,
            optimizer=optimizer
        )

        runner.infer(
            loaders= loaders,
            callbacks= callbacks,
            verbose= True
        )
        count = 0
        for i, (input, output) in enumerate(zip(self.__data, callbacks['infer'].predictions['logits'])):
            threshold = 0.5
            image, mask = input['image'], input['mask']
            rez = self.__sigmoid(output[0].copy())
            output = (output > threshold).astype(np.uint8) * 255.
            cv2.imwrite(os.path.join(self.__rez_dir, f'resMask{i}.png'), output[0])

if __name__ == '__main__':
    infer = Infer('data', 'logs100', 'results100', 2)
    infer.inference()
