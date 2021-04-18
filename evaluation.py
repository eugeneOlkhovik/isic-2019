import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from networks.efficientnet import get_EffNet

from utils import reseed, get_writers, get_data_loaders
from performance import PerformanceTracker
from config import Config, ConfigTwoClasses


chkpt_dir = os.path.join('ensemble_effnet', 'ensemble_1')

arch_to_chkpt_subdir= {
    'efficientnet-b0': 'adabelief__effnet__efficientnet_b0__lr__0_0006',
    'efficientnet-b1': 'adabelief__effnet__efficientnet_b1__lr__0_0006',
    'efficientnet-b2': 'adabelief__effnet__efficientnet_b2__lr__0_0002',
    'efficientnet-b3': 'adabelief__effnet__efficientnet_b3__lr__0_0003',
    'efficientnet-b4': 'adabelief__effnet__efficientnet_b4__lr__0_0003',
}


def load_model(cfg, model_arch, chkpt_subdir):
    checkpoint_path = os.path.join(chkpt_dir, chkpt_subdir, 'model.pth.tar')
    model = get_EffNet(cfg, model_arch)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():

    cfg = ConfigTwoClasses()
    reseed(cfg.SEED)

    batch_size = 16
    train_loader, val_loader = get_data_loaders(cfg, batch_size)

    ns = 'evaluation_ns'
    evaluation_dir = os.path.join('evaluation', ns)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)


    table_writer = SummaryWriter(evaluation_dir + '/logs/' + 'hparam' + '/scalars/', max_queue=1000)

    for model_arch, chkpt_subdir in arch_to_chkpt_subdir.items():
        print('\n')

        model = load_model(cfg, model_arch, chkpt_subdir)
        tag = chkpt_subdir

        writers = get_writers(evaluation_dir, tag)
        epoch = 1

        pt = PerformanceTracker(model, cfg, val_loader, writers, epoch)
        pt.forward()
        pt.compute_performance_metrics()
        pt.drop_pred_images()
        pt.drop_cam_images()
        pt.drop_conf_mat_images()
        pt.drop_pr_image()
        pt.drop_roc_image()

        pm = pt.perf_metrics
        table_writer.add_hparams(hparam_dict={'name': model_arch},
                                 metric_dict={'bal_acc': pm.bal_acc,
                                              'precision': pm.precision,
                                              'recall': pm.recall,
                                              'f1_score': pm.f1_score,
                                            #   'hparam/roc_auc_score': pm.roc_auc_score,                                            
                                              })

        for writer in writers.values():
            writer.flush()
            writer.close()



if __name__ == '__main__':
    main()
