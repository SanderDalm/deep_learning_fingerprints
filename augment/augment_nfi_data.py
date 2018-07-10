import os
from os import path
from glob import glob

import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from tqdm import tqdm

import config
from augment.augment_fingerprint import augment_fingerprint

DATAPATH = path.join(config.datadir, 'NFI')

images = glob(DATAPATH+'*/BMP/*')

for filename in tqdm(images):

    img = rgb2gray(imread(filename))

    for aug_number in [1, 2, 3, 4]:

        aug = augment_fingerprint(img)

        if not path.exists(DATAPATH + '/Aug{}'.format(aug_number)):
            os.makedirs(DATAPATH + '/Aug{}'.format(aug_number))
        np.save(DATAPATH+'/Aug{}/{}'.format(aug_number, filename.split('/')[-1]), aug)

# import matplotlib.pyplot as plt
# filename = images[0]
# orig = rgb2gray(imread(filename))
# aug1 = np.load(filename.replace('BMP', 'Aug1')+'.npy')
# aug2 = np.load(filename.replace('BMP', 'Aug2')+'.npy')
# aug3 = np.load(filename.replace('BMP', 'Aug3')+'.npy')
# plt.imshow(np.concatenate([orig, aug1, aug2, aug3]), cmap='gray')
