"""This module has some useful functions"""

import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    elif not os.path.exists(paths):
        os.makedirs(paths)

def compare_images_colab(real_img, generated_img, data, threshold=0.4):
    diff_img = np.abs(generated_img - real_img)
    threshold = threshold * 256.
    diff_img[diff_img <= threshold] = 0

    anomaly_img = np.zeros_like(real_img)
    #anomaly_img[:, :, :] = real_img
    anomaly_img[np.where(diff_img>0)[0], np.where(diff_img>0)[1]] = [0, 0, 200]

    return anomaly_img
