import torch

class Config:

    # Image properties
    scale_factor = 1
    IMG_HEIGHT = int(256 / scale_factor)
    IMG_WIDTH = int(256 / scale_factor)
    IMG_CHANNELS = 3

    # Learning process parameters
    SEED = 1863
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Gradient class activation maps
    feature_module = 'layer4'
    target_layer_names = ['2']

    data_split = {'train': 0.7, 'val': 0.29, 'test':0.01}
    class_names = [
        'Melanoma',
        'Melanocytic nevus',
        'Basal cell carcinoma',
        'Actinic keratosis',
        'Benign keratosis',
        'Dermatofibroma',
        'Vascular lesion',
        'Squamous cell carcinoma',
        'None of the above',
    ]
    class_names_short = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC','SCC','UNK']

    # Sanity check
    assert len(class_names) == len(class_names_short)
    num_classes = len(class_names)

    # From Dataset metastudy
    ds_mean = [0.6674, 0.5294, 0.5240]
    ds_std = [0.2235, 0.2035, 0.2152]

    class_cases = [4522, 12875, 3323, 867, 2624, 239, 253, 628, 0]
    total_cases = sum(class_cases)
    # class_weights = [x/total_cases for x in class_cases]
    class_weights = []
    for x in class_cases:
        class_weights += [x / total_cases]

    # Data specification
    # root_dir = '/media/vpad/Новый том/ISIC_2019_Training_Input/'
    root_dir = '/Users/eugeneolkhovik/python_files/ML/melanoma/archive'
    image_dir = 'ISIC_2019_Training_Input/ISIC_2019_Training_Input'
    gt_table = 'extracted_MEL_BCC.csv'
    meta_file = 'ISIC_2019_Training_Metadata.csv'

class ConfigTwoClasses(Config):
    gt_table = 'extracted_MEL_BCC.csv'

    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    class_names = ['Melanoma', 'BCC']
    class_names_short = ['MEL', 'BCC']
    class_weights = [1, 1]
    num_classes = len(class_names)


if __name__ == "__main__":

    cfg = Config()

    from IPython import embed
    embed()