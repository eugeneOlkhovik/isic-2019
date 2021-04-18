import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data_loader import DataManager


def get_data_loaders(cfg, args):
    data_man = DataManager(cfg)
    train_loader = DataLoader(data_man.train_ds,
                              batch_size=args.batch_size,
                              num_workers=cfg.NUM_WORKERS,
                              pin_memory=cfg.PIN_MEMORY,
                              shuffle=True)

    val_loader = DataLoader(data_man.val_ds,
                            batch_size=args.batch_size,
                            num_workers=cfg.NUM_WORKERS,
                            pin_memory=cfg.PIN_MEMORY,
                            shuffle=False)
    return train_loader, val_loader


def get_writers(wr_root, tag):
    writers = {
        'scalar': SummaryWriter(wr_root + '/logs/' + tag + '/scalars/', max_queue=1000),
        'predicts': SummaryWriter(wr_root + '/logs/' + tag + '/predicts/', max_queue=1000),
        'conf_mat': SummaryWriter(wr_root + '/logs/' + tag + '/conf_mat/', max_queue=1000),
        'gradcam': SummaryWriter(wr_root + '/logs/' + tag + '/gradcam/', max_queue=1000),
        'pr_curve': SummaryWriter(wr_root + '/logs/' + tag + '/pr_curve/', max_queue=1000),
        'roc_curve': SummaryWriter(wr_root + '/logs/' + tag + '/roc_curve/', max_queue=1000),
    }
    return writers


def save_checkpoint(net, epoch, loss, optimizer, filename='my_model.pth.tar'):
    print('=> Saving checkpoint')
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def reseed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
