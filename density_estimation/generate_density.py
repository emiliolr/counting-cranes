import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial
import scipy.ndimage.filters

#This is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
#  seems to be pretty inefficient... would definitely want to save these instead of generating on the fly!
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype = np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize = leafsize)
    distances, locations = tree.query(pts, k = 4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype = np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) // 2. // 2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode = 'constant')
    print ('done.')

    return density

if __name__ == '__main__':
    #TESTING gaussian_filter_density:
    import PIL.Image as Image

    import sys
    sys.path.append('/Users/emiliolr/Desktop/counting-cranes')
    from utils import *

    single_annot_fp = '/Users/emiliolr/Desktop/Conservation Research/final_dataset/annotations/FLIR2_20210321_201851_358_2510.xml'
    single_img_fp = '/Users/emiliolr/Desktop/Conservation Research/final_dataset/images/FLIR2_20210321_201851_358_2510.TIF'

    bboxes = get_bboxes(single_annot_fp)
    points = get_points(bboxes) #this is a list of lists, w/[x, y] coords for points
    img = np.array(Image.open(single_img_fp))

    point_matrix = np.zeros((img.shape[0], img.shape[1]))
    for p in points:
        point_matrix[p[1], p[0]] = 1

    density = gaussian_filter_density(point_matrix)
