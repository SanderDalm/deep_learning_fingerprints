import os
from os import path
from glob import glob

import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from tqdm import tqdm

import config
from augment.augment_fingerprint import augment_fingerprint, generate_perlin_noise

for n in range(0, 8, 1):

    print(n)
    DATAPATH = path.join(config.datadir, 'sd04/png_txt/figs_{}'.format(n))
    images = glob(DATAPATH+'/*.png')
    #perlin_noise = generate_perlin_noise((512, 512))

    for aug_number in [1, 2, 3, 4]:
        print('    ' + str(aug_number))
        for filename in tqdm(images):

            img = rgb2gray(imread(filename))

            #if np.random.normal() < .05:
            #    perlin_noise = generate_perlin_noise((512, 512))
            aug = augment_fingerprint(img)
            #aug = np.minimum(perlin_noise, aug)

            if not path.exists(DATAPATH + '/Aug{}'.format(aug_number)):
                os.makedirs(DATAPATH + '/Aug{}'.format(aug_number))
            np.save(DATAPATH+'/Aug{}/{}'.format(aug_number, filename.split('/')[-1]), aug)

import matplotlib.pyplot as plt
filename = images[0]
orig = rgb2gray(imread(filename))
plt.imshow(orig, cmap='gray')
plt.show()

aug = augment_fingerprint(orig)
plt.imshow(aug, cmap='gray')
plt.show()

aug1 = np.load(filename.replace('BMP', 'Aug1')+'.npy')
aug2 = np.load(filename.replace('BMP', 'Aug2')+'.npy')
aug3 = np.load(filename.replace('BMP', 'Aug3')+'.npy')
plt.imshow(np.concatenate([orig, aug1, aug2, aug3], axis=1), cmap='gray')