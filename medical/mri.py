import numpy
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from skimage.filters import threshold_otsu
from utils.digital_image import largest_connected_component


def mri_mask(image: numpy.ndarray):
    threshold = threshold_otsu(image, 32) / 4
    mask = binary_fill_holes(image > threshold)
    mask = binary_erosion(mask, iterations=1)
    mask = largest_connected_component(mask)
    mask = binary_dilation(mask, iterations=3)
    return mask & (image>0)