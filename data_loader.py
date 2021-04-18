import os
import numpy as np
import pandas as pd

import cv2
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
from torch.utils.data import Dataset

class DataManager():
    def __init__ (self, cfg):

        self.cfg = cfg
        gt_path = os.path.join(self.cfg.root_dir, self.cfg.gt_table)
        self.gt_table = pd.read_csv(gt_path)
        self.gt_table = self.gt_table.drop(columns=['Unnamed: 0'])

        self.class_names = list(self.gt_table.columns)[1:]
        image_dir = os.path.join(self.cfg.root_dir, self.cfg.image_dir)

        self.data = []
        self.image_paths = []
        for index, row in self.gt_table.iterrows():
            image_file = os.path.join(image_dir, row['image']) + '.jpg'
            label = list(row == 1.0).index(True)
            label -= 1
            # label = list(row[1:])
            self.data.append([image_file, label])
            self.image_paths.append(index)
            from IPython import embed
            embed()
        # Split data according to the config setup
        # NOTE data_split = {'train': 0.6, 'val': 0.2, 'test':0.2}
        num_intances = len(self.data)
        train_last_ind = int(num_intances * self.cfg.data_split['train'])
        val_last_ind = int(num_intances * (self.cfg.data_split['train'] + self.cfg.data_split['val']))

        self.train_data = self.data[:train_last_ind]
        self.val_data = self.data[train_last_ind:val_last_ind]
        self.test_data = self.data[val_last_ind:]

        # self.train_data = self.data[:16]
        # self.val_data = self.data[:16]
        # self.test_data = self.data[:16]

        # Compose trf and create dss
        self.train_trf, self.val_trf = compose_im_trf(cfg)
        self.train_ds = ImageFolder(self.train_data, self.cfg, self.train_trf)
        self.val_ds = ImageFolder(self.val_data, self.cfg, self.val_trf)
        self.test_ds = ImageFolder(self.test_data, self.cfg, self.val_trf)


class ImageFolder(Dataset):
    def __init__(self, data, cfg, transform=None):
        super(ImageFolder, self).__init__()
        self.data = data
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        image = np.array(Image.open(img_file))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        return image, label

    def load_image(self, index):
        image = Image.open(self.data[index][0])
        image = image.resize((self.cfg.IMG_WIDTH, self.cfg.IMG_HEIGHT))
        return np.array(image)
    
    def load_image_orig_shape(self, index):
        image = Image.open(self.data[index][0])
        return np.array(image)


def compose_im_trf(cfg):
    transform = A.Compose([
        A.Resize(width=cfg.IMG_WIDTH, height=cfg.IMG_HEIGHT),
        # A.RandomCrop(width=cfg.IMG_WIDTH, height=cfg.IMG_HEIGHT),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20,
                   b_shift_limit=20, p=0.9),
        A.OneOf([
                A.Blur(blur_limit=1, p=0.5),
                A.ColorJitter(p=0.5),
                ], p=1.0),
        A.Normalize(
            mean=cfg.ds_mean,
            std=cfg.ds_std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    transform_val = A.Compose([
        A.Resize(width=cfg.IMG_WIDTH, height=cfg.IMG_HEIGHT),
        A.Normalize(
            mean=cfg.ds_mean,
            std=cfg.ds_std,
            # mean=[0.0, 0.0, 0.0],
            # std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    return transform, transform_val

def compose_test_trf(cfg):
    transform_test = A.Compose([
       A.Normalize(
            mean=cfg.ds_mean,
            std=cfg.ds_std,
            max_pixel_value=255.0,
        ),
       ToTensorV2(),
    ])
    return transform_test


if __name__ == "__main__":

    from config import Config

    cfg = Config()
    data_man = DataManager(cfg)

    print('len(train_ds.data)', len(data_man.train_ds.data))
    print('len(val_ds.data)', len(data_man.val_ds.data))
    print('len(test_ds.data)', len(data_man.test_ds.data))

    from IPython import embed
    embed()
