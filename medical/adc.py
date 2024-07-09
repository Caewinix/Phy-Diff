import numpy
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
from skimage.filters import threshold_otsu
from utils.digital_image import largest_connected_component

class ArgumentError(Exception):
    r"""Thrown by an application when an invalid command line argument has been supplied.
    """
    pass

def adc_map(high_b_image: numpy.ndarray, base_b_image: numpy.ndarray, high_b, base_b=0, threshold=None):
    # convert to float
    base_b_image = base_b_image.astype(float)
    high_b_image = high_b_image.astype(float)
    # check if supplied threshold value as well as the b value is above 0
    if threshold is not None and not threshold >= 0:
        raise ArgumentError('The supplied threshold value must be greater than 0, otherwise a division through 0 might occur.')
    if not high_b > 0:
        raise ArgumentError('The supplied b-value must be greater than 0.')
    # compute threshold value if not supplied
    if threshold is None:
        b0thr = threshold_otsu(base_b_image, 32) / 4. # divide by 4 to decrease impact
        bxthr = threshold_otsu(high_b_image, 32) / 4.
        if 0 >= b0thr:
            raise ArgumentError('The supplied b0image seems to contain negative values.')
        if 0 >= bxthr:
            raise ArgumentError('The supplied bximage seems to contain negative values.')
    else:
        b0thr = bxthr = threshold
    # threshold b0 + bx DW image to obtain a mask
    # b0 mask avoid division through 0, bx mask avoids a zero in the ln(x) computation
    mask = binary_fill_holes(base_b_image > b0thr) & binary_fill_holes(high_b_image > bxthr)
    # perform a number of binary morphology steps to select the brain only
    mask = binary_erosion(mask, iterations=1)
    mask = largest_connected_component(mask)
    mask = binary_dilation(mask, iterations=3)
    # compute the ADC
    adc = numpy.zeros(base_b_image.shape, base_b_image.dtype)
    mask = mask & (high_b_image>0) & (base_b_image>0)
    adc[mask] = -1. * numpy.log(high_b_image[mask] / base_b_image[mask]) / (high_b - base_b)
    adc[adc < 0] = 0
    return adc

def masked_adc_map(high_b_image: numpy.ndarray, high_b_mask, base_b_image: numpy.ndarray, base_b_mask, high_b, base_b=0):
    mask = high_b_mask & base_b_mask
    adc = numpy.zeros(base_b_image.shape, base_b_image.dtype)
    adc[mask] = -1. * numpy.log(high_b_image[mask] / base_b_image[mask]) / (high_b - base_b)
    adc[adc < 0] = 0
    return adc, mask