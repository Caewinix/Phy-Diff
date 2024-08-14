from scipy.ndimage import label
import numpy

def largest_connected_component(img, structure = None):
    labeled_array, num_features = label(img, structure)
    component_sizes = [numpy.count_nonzero(labeled_array == label_idx) for label_idx in range(1, num_features + 1)]
    if len(component_sizes) > 0:
        largest_component_idx = numpy.argmax(component_sizes) + 1
    else:
        largest_component_idx = 0
    out = numpy.zeros(img.shape, numpy.bool_)
    out[labeled_array == largest_component_idx] = True
    return out