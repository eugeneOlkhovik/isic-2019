import os

os.system('cd ~/Code/derma_classifier')


train_setups = [
    '--arch effnet --arch_ver efficientnet-b0 --ns ensemble_1 --lr 6e-4 --batch_size 32',
    '--arch effnet --arch_ver efficientnet-b1 --ns ensemble_1 --lr 6e-4 --batch_size 32',
    '--arch effnet --arch_ver efficientnet-b2 --ns ensemble_1 --lr 2e-4 --batch_size 16',
    '--arch effnet --arch_ver efficientnet-b3 --ns ensemble_1 --lr 3e-4 --batch_size 16',
    '--arch effnet --arch_ver efficientnet-b4 --ns ensemble_1 --lr 3e-4 --batch_size 16',
]


for trs in train_setups:
    cmd = 'python3 train.py ' + trs + ' --total_epoch 30'
    print('\n Traning next model from assebly with')
    print(cmd, '\n')
    os.system(cmd)
