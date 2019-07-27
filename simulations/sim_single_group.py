import sys, os
sys.path.append('../')
import torch
from PIL import Image
from torchvision.transforms import transforms
from Att_Weight_KPN import Att_Weight_KPN
from KPN import KPN
import numpy as np

images_dir = ''
if not os.path.isdir(images_dir):
    raise ValueError('Path should be a directory.')

files = os.listdir(images_dir)
cnt_files = len(files)
rand_index = np.random.permutation(cnt_files)[:8]
