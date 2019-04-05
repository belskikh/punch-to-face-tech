import os
import collections
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from catalyst.contrib.models import ResNetUnet
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import (InferCallback, CheckpointCallback)
from utils import x_to_torch, y_to_torch
from dataset import VehicleDataset

class Infer:
    def __init__(self, csv_file, logs_dir, rez_dir, batch_size):
        super(Infer, self).__init__()
        self.__csv_file = csv_file
        self.__logs_dir = logs_dir
        if os.path.exists(rez_dir) == False:
            os.mkdir(rez_dir)
            self.__rez_dir = rez_dir
        else:
            self.__rez_dir = rez_dir
        self.__batch_size = batch_size
        self.__data = None

    def __get_data(self):
        data = VehicleDataset(self.csv_file)
        loaders = collections.OrderedDict()
        loader = DataLoader(
            dataset=data,
            batch_size=self.__batch_size,
            shuffle=True
        )
        loaders['infer'] = loader
        return loaders

    def inference(self):
        model = ResNetUnet(num_classes=3, num_filters=64)
        loaders = self.__get_data()
        runner = SupervisedRunner()
        runner.infer(
            model=model,
            loaders=loaders,
            callbacks=[
                CheckpointCallback(
                    resume=os.path.join(self.__logs_dir, 'checkpoints/best.pth')
                )
            ]
        )
        count = 0
        for i, (input, output) in enumerate(zip(self.__data, runner.callbacks[1].predictions['logits'])):
            threshold = 0.5
            output = torch.sigmoid(output)
            image, mask = input['image'], input['mask']
            car = (output[0] > threshold).astype(np.uint8) * 255.
            bus = (output[1] > threshold).astype(np.uint8) * 255.
            track = (output[2] > threshold).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(self.__rez_dir, 'frames', f'frame{i}.jpg'), image)
            cv2.imwrite(os.path.join(self.__rez_dir, 'car', f'resMask{i}.png'), car)
            cv2.imwrite(os.path.join(self.__rez_dir, 'bus', f'resMask{i}.png'), bus)
            cv2.imwrite(os.path.join(self.__rez_dir, 'track', f'resMask{i}.png'), track)


