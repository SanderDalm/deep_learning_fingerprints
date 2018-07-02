import noise
import numpy as np
from scipy.misc import toimage
from tqdm import tqdm

import config

shape = (512, 512)
scale = 10.0  # 10
octaves = 3
persistence = 0.2
lacunarity = 2.0
pnoise = np.zeros(shape)

for n in tqdm(range(10)):
    for i in range(shape[0]):
        for j in range(shape[1]):
            pnoise[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0)

    pnoise[pnoise < 0] = 0
    #toimage(pnoise).show()

    np.save(config.datadir + '/perlin_noise/{}'.format(n), pnoise)


arr = np.load(config.datadir + '/perlin_noise/4.npy')

#import matplotlib.pyplot as plt
#plt.imshow(arr)
#plt.show()