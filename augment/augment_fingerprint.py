import noise
import numpy as np
from scipy.misc import toimage

from skimage.io import imread
import cv2
from skimage.morphology import remove_small_objects
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT, borderValue=1.0)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='constant', cval=1.0).reshape(shape)


def generate_perlin_noise(shape, perlin_scale=10.0, perlin_octaves=6, perlin_persistence=0.5, perlin_lacunarity=2.0):
    pattern = np.zeros(shape)
    z = np.random.randint(0, 65353)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pattern[i][j] = noise.pnoise3(i / perlin_scale,
                                          j / perlin_scale,
                                          z,
                                          octaves=perlin_octaves,
                                          persistence=perlin_persistence,
                                          lacunarity=perlin_lacunarity,
                                          repeatx=1024,
                                          repeaty=1024,
                                          base=0)

    # apply a threshold on snortjes
    snotjes_threshold = 0.2
    pattern[pattern < snotjes_threshold] = snotjes_threshold

    # scale snotjes between 0 and 1
    pattern = np.interp(pattern, (pattern.min(), pattern.max()), (0, 1))

    # remove small snotjes
    mask = pattern > 0
    mask = remove_small_objects(mask, min_size=64)
    pattern = pattern * mask
    pattern = 1.0 - pattern

    return pattern

def augment_fingerprint(fingerprint):
    fingerprint_aug = elastic_transform(fingerprint, fingerprint.shape[1] * 3, fingerprint.shape[1] * 0.1, fingerprint.shape[1] * 0.1)
    return fingerprint_aug


