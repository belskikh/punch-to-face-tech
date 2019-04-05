import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from catalyst.contrib.models import ResNetUnet
from tiles import ImageSlicer

def predict(logs_dir, tiles):
    model = ResNetUnet(num_classes=3, num_filters=64)
    model.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    state = torch.load(f'{logs_dir}/checkpoints/best.pth')
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    cars_tile = []
    bus_tile = []
    truck_tile = []
    for i, tile in enumerate(tiles):
        img = transform(tile).cuda()
        output = model(img.unsqueeze_(0))
        print(f'Tile {i} predicted!')
        out = torch.sigmoid(output)
        print(out.shape)
        pred = np.moveaxis(out.data.cpu().numpy(), 0, -1)
        print(pred.shape)
        car = (pred[0] > 0.2).astype(np.uint8) * 255.
        print(car.shape)
        cars_tile.append(car)
        bus = (pred[1] > 0.5).astype(np.uint8) * 255.
        bus_tile.append(bus)
        truck = (pred[2] > 0.5).astype(np.uint8) * 255.
        truck_tile.append(truck)
    return np.array([cars_tile, bus_tile, truck_tile])

def overlay_on_frame(frame, mask, color):
    assert frame.shape[0] ==  mask.shape[0]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, color, 2)
    return frame

def main():
    video = sys.argv[1]
    res_folder = sys.argv[2]
    if os.path.exists(res_folder) == False:
        os.mkdir(res_folder)
    cap = cv2.VideoCapture(video)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print(frame.shape)
        slicer = ImageSlicer(frame.shape, tile_size=[512, 512], tile_step=[150,150])
        print('Slicer ready!')
        tiles = slicer.split(frame)
        print(f'frame {count} was splited into tiles!')
        out  = predict('finalLogs', tiles)
        print(out.shape)
        print(out[0].shape)
        car = slicer.merge(out[0])
        bus = slicer.merge(out[1])
        truk = slicer.merge(out[2])
        rez = overlay_on_frame(frame, car, (0, 0, 255))
        rez = overlay_on_frame(rez, bus, (0, 255, 0))
        rez = overlay_on_frame(rez, truck, (255, 0, 0))
        cv2.imwrite('{}/{}.jpg'.format(res_folder, count), rez)
        print(f'Res frame-{count} was saved!')
        count += 1

if __name__ == '__main__':
    main()
