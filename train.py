import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

from networks.efficientnet import get_EffNet
from adabelief_pytorch import AdaBelief

from utils import save_checkpoint, reseed, get_writers, get_data_loaders
from performance import PerformanceTracker
from config import Config, ConfigTwoClasses


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--total_epoch', default=100, type=int, help='Total number of training epochs')
    parser.add_argument('--decay_epoch', default=75, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--arch', default='effnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vgg', 'effnet'])
    parser.add_argument('--arch_ver', default='efficientnet-b0', type=str,
                        help='architecture version of the choosen model')
    parser.add_argument('--optim', default='adabelief', type=str, help='optimizer',
                        choices=['sgd', 'adam', 'adamw', 'adabelief'])
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')

    parser.add_argument('--eps', default=1e-16, type=float, help='eps for var adam')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')
    parser.add_argument('--ns', default='my_model_ns', type=str,
                        help='namespace for model checkouts and summaries')
    parser.add_argument('--load_model', default=False, type=bool,
                        help='whether load model')
    return parser


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgd':
        return torch.optim.SGD(model_params, args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return torch.optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adamw':
        return torch.optim.AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adabelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                         weight_decay=args.weight_decay, eps=args.eps,
                         print_change_log=False)
    else:
        print('Optimizer not found')


def train_fn(loader, model, optimizer, scheduler, loss_fn, scaler, device):

    loop = tqdm(loader)
    losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.long())

        losses.append(loss)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm progress bar
        loss_val = loss.item()
        loop.set_postfix(loss=loss_val)

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

    return loss_val


def main():

    parser = get_parser()
    args = parser.parse_args()

    tag = '__'.join([args.optim, args.arch, args.arch_ver, 'lr', str(args.lr)])
    now = datetime.now()
    tag += now.strftime("%H_%M_%S")
    tag = tag.replace('-', '_')
    tag = tag.replace('.', '_')

    experiments_dir = os.path.join('experiments', args.ns)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)

    model_dir = os.path.join(experiments_dir,  tag)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    writers = get_writers(experiments_dir, tag)
    model_save_path = os.path.join(model_dir, 'model.pth.tar')

    cfg = ConfigTwoClasses()
    reseed(cfg.SEED)

    train_loader, val_loader = get_data_loaders(cfg, args.batch_size)

    # model = Net(num_classes=cfg.num_classes).to(cfg.DEVICE)
    model = get_EffNet(cfg, args.arch_ver)
    optimizer = create_optimizer(args, model.parameters())
    weights = torch.tensor(cfg.class_weights, device=cfg.DEVICE, dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True)

    if args.load_model:
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Model loaded with last epoch of {last_epoch} and loss {loss}')
    else:
        last_epoch = 0


    writer_scalar = writers['scalar']

    for epoch in range(last_epoch, args.total_epoch):
        loss = train_fn(train_loader, model, optimizer, lr_scheduler, loss_fn, scaler, cfg.DEVICE)
        writer_scalar.add_scalar('Training / Loss', loss, global_step=epoch)

        pt = PerformanceTracker(model, cfg, val_loader, writers, epoch)
        pt.forward()
        pt.check_accuracy(epoch)

        if epoch % 2 == 0:
            save_checkpoint(model, epoch, loss, optimizer, model_save_path)

        if epoch % 5 == 0:
            pt.drop_visuals_to_tboard()

        for writer in writers.values():
            writer.flush()

    for writer in writers.values():
        writer.close()



if __name__ == '__main__':
    main()
