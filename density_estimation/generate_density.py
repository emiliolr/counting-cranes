import sys
sys.path.append('/Users/emiliolr/Desktop/counting-cranes')
from utils import *

import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial
import scipy.ndimage.filters

#This is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet w/tweaks from Gao et al. (2020)
#  seems to be pretty inefficient... would definitely want to save these instead of generating on the fly!
def adaptive_gaussian_density(gt):

    """
    Produces the ground truth density from point annotations using a gaussian filter.
    Inputs:
      - gt: the ground truth point annotations as a [height x width] numpy array w/1s at the points
    Outputs:
      - The ground truth density as a numpy array
    """

    density = np.zeros(gt.shape, dtype = np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize = leafsize)
    distances, locations = tree.query(pts, k = 4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype = np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) // 2. // 2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode = 'constant')

    return density

def fixed_gaussian_density(gt, sigma = 1):

    """
    A wrapper for scipy's fixed gaussian filter.
    Inputs:
      - gt: the ground truth point annotations as a [height x width] numpy array w/1s at the points
      - sigma: the sigma to use for the fixed filter
    Outputs:
      - The ground truth density as a numpy array
    """

    return scipy.ndimage.filters.gaussian_filter(gt, sigma = sigma)

def density_from_bboxes(bboxes, image, filter_type = 'adaptive', sigma = None):

    """
    Convenience function to ranslate bounding boxes to a density.
    Inputs:
      - bboxes: a list of lists containing all bboxes for the image
      - image: the image corresponding to the annotations (a tensor)
      - filter_type: either fixed or adaptive
      - sigma: the sigma to use (only for a filter_type of fixed)
    Outputs:
      - The ground truth density as a numpy array
    """

    points = get_points(bboxes) #this is a list of lists, w/[x, y] coords for points

    point_matrix = np.zeros((image.shape[1], image.shape[2]))
    for p in points:
        point_matrix[p[1], p[0]] = 1 #place a one at each point annotation

    if filter_type == 'adaptive':
        return adaptive_gaussian_density(point_matrix)
    elif filter_type == 'fixed':
        assert sigma is not None, 'Please provide a sigma for the fixed filter'
        return fixed_gaussian_density(point_matrix, sigma = sigma)

if __name__ == '__main__':
    #TESTING density_from_bboxes:
    import PIL.Image as Image

    import sys
    sys.path.append('/Users/emiliolr/Desktop/counting-cranes')
    from utils import *

    single_annot_fp = '/Users/emiliolr/Desktop/Conservation Research/final_dataset/annotations/FLIR2_20210321_201851_358_2510.xml'
    single_img_fp = '/Users/emiliolr/Desktop/Conservation Research/final_dataset/images/FLIR2_20210321_201851_358_2510.TIF'

    bboxes = get_bboxes(single_annot_fp)
    image = np.array(Image.open(single_img_fp))
    density = density_from_bboxes(bboxes, image, filter_type = 'fixed', sigma = 1.5)
    print(density.shape)
