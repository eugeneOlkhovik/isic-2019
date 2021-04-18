import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import glob
import pandas as pd
import re
import csv
import argparse
import cv2

from networks.efficientnet import get_EffNet
from utils import save_checkpoint, reseed, load_model
from train import get_parser, get_data_loaders
from config import Config, ConfigTwoClasses
from data_loader import DataManager, ImageFolder, compose_im_trf, compose_test_trf
from preprocessing import get_image_patches


image_path = '/Users/eugeneolkhovik/python_files/ML/melanoma/archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
experiments_dir = 'experiments'
ensemble_dir = 'ensemble_1'


chkpt_dir = os.path.join('ensemble_effnet', 'ensemble_1')

arch_to_chkpt_subdir= {
    'efficientnet-b0': 'adabelief__effnet__efficientnet_b0__lr__0_0006',
    'efficientnet-b1': 'adabelief__effnet__efficientnet_b1__lr__0_0006',
    'efficientnet-b2': 'adabelief__effnet__efficientnet_b2__lr__0_0002',
    'efficientnet-b3': 'adabelief__effnet__efficientnet_b3__lr__0_0003',
    'efficientnet-b4': 'adabelief__effnet__efficientnet_b4__lr__0_0003',
}



    
    
class PredictionWriter:
     # load image fucn do not convert to tensor 
     # load model fucn 
     # get patches of loaded image and add default image to dict 
     # from each model of ensemble make predict for each image and write in list of dict
     # write to csv (image, b_(i)_class(j), b_(i)_class(j)_patch_(k), class_(j))
               # b0_tl_BCC, b0_tl_ML
    def __init__(self, cfg, data_loader, pass_to_csv, patch_size=(256, 256)): 
        self.cfg = cfg
        self.data_loader = data_loader
        self.pass_to_csv = pass_to_csv
        self.patch_size = patch_size
        self.transform = compose_test_trf(self.cfg)
        self.data_frame = create_dataframe()
        
        
    def classify_image(self, model, patch_dict):
        patch_to_probas = {}
        for k_patch, patch_img  in patch_dict.items():
            augmentations = self.transform(image=patch_img)
            patch = augmentations["image"]
            output = model(patch)
            proba = torch.nn.functional.softmax(output, dim=1)[0]
            patch_to_probas[k_patch] = proba
            
        return patch_to_probas 
    
    def create_dataframe(self, column_sub_names):
        indexes = self.data_loader.image_paths
        columns = ['imageName']
        for model_name in arch_to_chkpt_subdir.keys():
            for img_patch in column_sub_names:
                for class_name in self.cfg.class_names_short:
                    column_name = re.search('b[0-9]', model_name).group(0)+ "_" + img_patch+ "_" + class_name
                    columns.append(column_name)
            columns.append("true_class")            
        return pd.DataFrame( index = indexes, columns= columns)
        
       
    # b0_tl_BCC, b0_tl_ML
    def image_preds_to_dataFrame(self, patch_name_to_pred_val,arch , image_name):
        for k_pred, pred in patch_name_to_pred_val.items():
            for i,class_name in enumerate(self.cfg.class_names_short):
                pred_column = arch + "_" + k_pred + "_" + class_name
                self.data_frame.loc[image_name, pred_column] = pred[i] 
            
        
            
        
    def run(self):
        for model_arch, chkpt_subdir in arch_to_chkpt_subdir.items():
            print('\n')
            model = load_model(cfg, model_arch, chkpt_subdir) #import from utils 
            for i in range(len(self.data_loader.dataset.data)):
                image = self.data_loader.dataset.load_image_orig_shape(i)
                image_name = self.data_loader.dataset.image_paths[i]
                
                image_patches =  get_image_patches(image, self.patch_size)
                image_patches['orig'] = cv2.resize(image, self.patch_size)
                
                img_probas = classify_image(model, image_patches)
                
                name_arch = re.search('b[0-9]', model_arch).group(0)
                image_preds_to_dataFrame(img_probas,name_arch, image_name)
                
        self.data_frame.to_csv('/Users/eugeneolkhovik/python_files/ML/melanoma/derma_classifier/meta_study/ensemble_pred.csv')  
        
    
if __name__ == "__main__":
    cfg = ConfigTwoClasses()
    data_manager = DataManager(cfg)
    _, data_loader = get_data_loaders(cfg,) # args 
    patches_name = ['tl', 'tr', 'bl', 'br', 'center', 'orig']
    
    writer = PredictionWriter()
    writer.create_dataframe(patches_name)
    writer.run()
    
    
"""
### experiment -> models directories

all_dirs = os.walk(experiments_dir) 
models_res = []
model_path = []
for i in next(all_dirs)[1]: 
    model_resualts_dir  = os.path.join(experiments_dir, i)
    models_res.append(model_resualts_dir)

### extract all files with pythorch extention 
for models in models_res:
    for i in glob.glob(os.path.join(models,'*/*.pth.tar')):
        #checkpoint = torch.load(i)
        #print(checkpoint.keys())                                        #dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'])
        #checkpoint.get('model_state_dict')
        #data = pd.DataFrame(checkpoint)
        #data.T.to_csv(path_or_buf=os.path.join(experiments_dir,'models_weights',re.search('b[0-5]', i).group(0) + ".csv"), header =True,index=False)
        model_path.append(i)
model_path = model_path.sort(key=lambda x:int(x.split()[0]))
"""
