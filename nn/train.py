import os
import collections
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from catalyst.contrib.models import ResNetUnet
from catalyst.contrib.criterion import FocalLoss
from catalyst.dl.experiments.runner import SupervisedRunner
from catalyst.dl.callbacks import (
    LossCallback, TensorboardLogger, OptimizerCallback, CheckpointCallback,  ConsoleLogger)
from dataset import VehicleDataset


class Model:
    def __init__(self, data_csv, logs_dir, batch_size, workers):
        self.data = data_csv
        if os.path.exists(logs_dir) == False:
            os.mkdir(logs_dir)
        self.logs_dir = logs_dir
        self.batch_size = batch_size
        self.workers = workers
        self.classes = ['car', 'bus', 'truck']

    def get_data(self):
        data = VehicleDataset(self.data)
        loaders = collections.OrderedDict()
        train_loader = DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers
        )
        valid_loader = DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )

        loaders['traint'] = train_loader
        loaders['valid'] = valid_loader

        return loaders

    def set_callbacks(self):
        callbacks = collections.OrderedDict()
        callbacks['loss'] = LossCallback()
        callbacks['optimizer'] = OptimizerCallback()
        callbacks['saver'] = CheckpointCallback()
        callbacks['logger'] = ConsoleLogger()
        callbacks['tflogger'] = TensorboardLogger()
        return callbacks

    def train(self, epoch):
        model = ResNetUnet(num_classes=3, num_filters=64)
        criterion = FocalLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        callbacks = self.set_callbacks()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)
        loaders = self.get_data()

        runner = SupervisedRunner()

        runner.train(
            model= model,
            criterion=criterion,
            loaders=loaders,
            logdir= self.logs_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=epoch
        )

if __name__ == '__main__':
    model = Model('data/data.csv', 'finalLogs', 1, 4)
    model.train(50)
