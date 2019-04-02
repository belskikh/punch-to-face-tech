import os
import csv
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import x_to_torch, y_to_torch, transforms

class VehicleDataset(Dataset):
    """
    3 Class Dataset:
    1 class: Cars
    2 class: Bus
    3 class: Trucks
    """
    def __init__(self, csv_file, transforms = True):
        super(VehicleDataset, self).__init__()
        self.data_frame = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx: int):
        targets = []
        image = cv2.imread(self.data_frame['image'][idx])
        car_mask = cv2.imread(self.data_frame['car'][idx], cv2.IMREAD_UNCHANGED)
        bus_mask = cv2.imread(self.data_frame['bus'][idx], cv2.IMREAD_UNCHANGED)
        truck_mask = cv2.imread(self.data_frame['truck'][idx], cv2.IMREAD_UNCHANGED)
        targets.append(image)
        targets.append(car_mask)
        targets.append(bus_mask)
        targets.append(truck_mask)
        if self.transforms:
            aug = transforms(targets)
        targets = torch.stack([y_to_torch(aug['mask']), y_to_torch(aug['mask1']), y_to_torch(aug['mask2'])])
        return {'features': x_to_torch(aug['image']), 'targets': targets}


