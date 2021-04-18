from albumentations.augmentations.transforms import ToFloat
from config import Config
from data_loader import DataManager, ImageFolder
import torch

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


def get_mean_std(loader, cfg):

    ch_sum, ch_sqrd_sum, n_batch = 0.0, 0.0, 0.0

    for data, classes in tqdm(loader):
        data = data.to(device=cfg.DEVICE)
        ch_sum += torch.mean(data, dim=[0, 2, 3])
        ch_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        n_batch += 1

    mean = ch_sum / n_batch
    std = (ch_sqrd_sum / n_batch - mean ** 2) ** 0.5

    return mean, std


def main():
        
    cfg = Config()
    data_man = DataManager(cfg)
    tf = A.Compose([
        A.Resize(width=cfg.IMG_WIDTH*2, height=cfg.IMG_HEIGHT*2),
        A.ToFloat(),
        ToTensorV2(),
    ])

    im_folder = ImageFolder(data_man.data, tf)

    loader = DataLoader(im_folder,
                        batch_size=64,
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=cfg.PIN_MEMORY,
                        shuffle=False)


    mean, std = get_mean_std(loader, cfg)

    print('mean', mean)
    print('std', std)


if __name__ == "__main__":
    main()
