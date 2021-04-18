
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from albumentations import Compose
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple


class ColorConstancy(ImageOnlyTransform):
    """Remove prevailing light from image by Shades of Gray algorithm.

    Args:
        norm_degree (int): the degree of norm, 6 is used in reference paper
        gamma (float): the value of gamma correction, 2.2 is used in reference paper

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, norm_degree=6, gamma=None, always_apply=True, p=1.0):
        super(ColorConstancy, self).__init__(always_apply, p)
        self.norm_degree = norm_degree
        self.gamma = gamma

    def apply(self, image, norm_degree=0, gamma=0,  **params):
        return shade_of_gray_cc(image, norm_degree, gamma)

    def get_params(self):
        return {
            "norm_degree": self.norm_degree,
            "gamma": self.gamma,
        }

    def get_transform_init_args_names(self):
        return ("norm_degree", "gamma")


def shade_of_gray_cc(input_img, norm_degree=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    norm_degree (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """

    img_dtype = input_img.dtype

    if gamma is not None:
        img = input_img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)
    else:
        img = input_img

    img = img.astype('float32')
    img_power = np.power(img, norm_degree)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/norm_degree)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)
    
    return img.astype(img_dtype)