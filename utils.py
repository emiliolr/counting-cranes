import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from torch import as_tensor
import os
import numpy as np
from itertools import chain

def get_bboxes(xml_fp):

    """
    Reads in all bounding boxes for an image.
    XML file is assumed to be in PascalVOC format.
    From https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python.
    Inputs:
      - xml_fp: an xml file w/bbox annotations
    Outputs:
      - A list of lists w/all bbox annotations in [xmin, ymin, xmax, ymax] format
    """

    tree = ET.parse(xml_fp)
    root = tree.getroot()

    width = int(root.findall('size')[0].findall('width')[0].text)
    height = int(root.findall('size')[0].findall('height')[0].text)

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        xmin = np.clip(int(float(boxes.find('bndbox/xmin').text)), 0, width) #using clip to ensure that bboxes are w/in the image
        ymin = np.clip(int(float(boxes.find('bndbox/ymin').text)), 0, height)
        xmax = np.clip(int(float(boxes.find('bndbox/xmax').text)), 0, width)
        ymax = np.clip(int(float(boxes.find('bndbox/ymax').text)), 0, height)
        list_with_all_boxes.append([xmin, ymin, xmax, ymax])

    return purge_invalid_bboxes(list_with_all_boxes)

def get_points(bboxes):

    """
    Builds point annotations from the centroid of bounding box annotations.
    Inputs:
      - bboxes: a list of lists containing bboxes in PascalVOC format
    Outputs:
      - A list of lists w/point annotations
    """

    points = []
    for b in bboxes:
        x_loc = int(0.5 * (b[0] + b[2]))
        y_loc = int(0.5 * (b[1] + b[3]))
        points.append([x_loc, y_loc])

    return points

def get_regression(bboxes):

    """
    Builds an image count from bboxes.
    Inputs:
      - bboxes: a list of lists containing bboxes in PascalVOC format
    Outputs:
      - A single integer for the image count
    """

    return len(bboxes)

def visualize_bboxes(image_fp, xml_fp, image = None, bboxes = None):

    """
    Draws bboxes on image.
    Will draw directly on provided image if bboxes and image are both provided.
    Inputs:
      - image_fp: the image filepath
      - xml_fp: the xml filepath
      - image: a PIL image
      - bboxes: a list of bboxes
    Outputs:
      - A PIL image w/bboxes drawn on
    """

    assert not((image is None and bboxes is not None) or (image is not None and bboxes is None)), 'Make sure to include both an image and bboxes'

    if image is None and bboxes is None: #if we were provided w/FPs, we'll have to pull these in..
        image = Image.open(image_fp)
        bboxes = get_bboxes(xml_fp)

    draw = ImageDraw.Draw(image)
    for b in bboxes:
        draw.rectangle(b, outline = 'red', width = 1)

    return image

def visualize_points(image_fp, xml_fp):

    """
    Draws point annotations on image.
    Inputs:
      - image_fp: the image filepath
      - xml_fp: the xml filepath
    Outputs:
      - A PIL image w/points drawn on
    """

    img = Image.open(image_fp)
    bboxes = get_bboxes(xml_fp)
    points = [tuple(coord) for coord in get_points(bboxes)]

    draw = ImageDraw.Draw(img)
    draw.point(points, fill = 'red')

    return img

def purge_invalid_bboxes(list_of_bboxes):

    """
    A convenience function to remove any invalid bounding boxes.
    Inputs:
     - list_of_bboxes: a list of lists that contains bboxes in [xmin, ymin, xmax, ymax] format
    Outputs:
      - A final list (or tensor) of valid bboxes
    """

    final_bboxes = []
    for box in list_of_bboxes:
        xmin, ymin, xmax, ymax = box

        if xmin < xmax and ymin < ymax: #conditions for a valid bbox!
            final_bboxes.append(box)

    return final_bboxes

def bbox_dataset_statistics(root_dir):

    """
    A function to get basic annotation statistics (average, minimum, maximum size, etc.).
    Inputs:
      - root_dir: the root directory for imagery/annotations
    Outputs:
      - A dictionary with several relevant bbox stats
    """

    annotation_fps = sorted(os.listdir(os.path.join(root_dir, 'annotations')))
    all_areas = []
    all_bbox_cts = []

    for fp in annotation_fps:
        annotation_fp = os.path.join(root_dir, 'annotations', fp)
        bboxes = get_bboxes(annotation_fp)
        areas = [(box[3] - box[1]) * (box[2] - box[0]) for box in bboxes]

        all_areas.append(areas) #the pixel areas for bird annotations in this image
        all_bbox_cts.append(len(bboxes)) #the number of birds in this image

    all_areas = list(chain(*all_areas))

    return_dict = {'avg_area' : np.mean(all_areas),
                   'min_area' : np.min(all_areas),
                   'max_area' : np.max(all_areas),
                   'avg_num_bboxes' : np.mean(all_bbox_cts),
                   'min_num_bboxes' : np.min(all_bbox_cts),
                   'max_num_bboxes' : np.max(all_bbox_cts),
                   'total_num_bboxes' : len(all_areas),
                   'num_annotated_imgs' : len(all_bbox_cts)}

    return return_dict

def bbox_counts_splits(root_dir, permutation, split):
    """
    A function to calculate bbox counts for each of a given train/val/test split
    Inputs:
      - root_dir: the root directory for imagery/annotations
      - permutation: the permutation of indices for the split
      - split: a 3-tuple containing the number of images in each split - format: (train, val, test)
    Outputs:
      - A dictionary with counts for each of the splits
    """

    annotation_fps = np.array(sorted(os.listdir(os.path.join(root_dir, 'annotations'))))
    annotation_fps_split = [annotation_fps[permutation[ : split[0]]],
                            annotation_fps[permutation[split[0] : split[0] + split[1]]],
                            annotation_fps[permutation[split[0] + split[1] : ]]]
    print(sorted(annotation_fps_split[2]))

    bbox_counts = {'train' : 0, 'val' : 0, 'test' : 0}

    for fps, split in zip(annotation_fps_split, ['train', 'val', 'test']):
        for fp in fps:
            annotation_fp = os.path.join(root_dir, 'annotations', fp)
            bboxes = get_bboxes(annotation_fp)

            bbox_counts[split] += len(bboxes)

    return bbox_counts

def pad_image(image, sides):

    """
    Adds the desired padding to the edges of the image.
    Inputs:
      - image: a PIL image
      - right, left, top, bottom: the number of pixels to zero pad on the original image
    Outputs:
      - A new PIL image w/the desired padding
    """

    right, left, top, bottom = sides
    width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom

    new_image = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    new_image.paste(image, (left, top))

    return new_image

def pad_parent_for_tiles(parent_image, tile_size = (224, 224)):

    """
    Pads a parent image so that it's large enough to accomodate an integer number of tiles.
    Inputs:
      - parent_image: the parent image to pad
      - tile_size: the size of the tile, in format (width, height)
    Outputs:
      - A padded PIL image
    """

    tile_width, tile_height = tile_size
    image_width, image_height = parent_image.size
    right_padding = tile_width - (image_width % tile_width)
    bottom_padding = tile_height - (image_height % tile_height)

    return pad_image(parent_image, (right_padding, 0, 0, bottom_padding))

def sort_AGL(observed_AGL):

    """
    A function to sort the data into our the closest (via absolute distance) AGL "bin" that we flew.
    Inputs:
      - observed_AGL: the actual AGL in feet
    Outputs:
      - The closest AGL among the "bins"
    """

    AGL_bins = np.array([2000, 2500, 3000, 3500, 4000, 4500, 5000])
    abs_dists = np.absolute(AGL_bins - observed_AGL)

    return AGL_bins[np.argmin(abs_dists)]

#TESTS:
if __name__ == '__main__':
    #TESTING purge_invalid_bboxes:
    dummy_boxes = [[0, 5, 0, 6], [1, 3, 2, 5], [44.0, 224.0, 48.0, 224.0]]
    # print(purge_invalid_bboxes(dummy_boxes))

    #TESTING bbox_dataset_statistics:
    import json

    config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
    DATA_FP = config['data_filepath_local']
    PERM = config['fixed_data_permutation']

    print(bbox_counts_splits(DATA_FP, PERM, (24, 4, 12)))
