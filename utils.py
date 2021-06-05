import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from torch import as_tensor
import os
import numpy as np

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

#TODO: may want to wait until after augmentations to translate - that way we can let albumentation handle bbox translation!
def get_points(xml_fp):

    """
    Builds point annotations from the centroid of bounding box annotations.
    Inputs:
      - xml_fp: an xml file w/bbox annotations
    Outputs:
      - A list of lists w/point annotations
    """

    bboxes = get_bboxes(xml_fp)

    points = []
    for b in bboxes:
        x_loc = int(0.5 * (b[0] + b[2]))
        y_loc = int(0.5 * (b[1] + b[3]))
        points.append([x_loc, y_loc])

    return points

def get_regression(xml_fp):

    """
    Builds an image count from bboxes.
    Inputs:
      - xml_fp: an xml file w/bbox annotations
    Outputs:
      - A single integer fro the image count
    """

    return len(get_bboxes(xml_fp))

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
    points = [tuple(coord) for coord in get_points(xml_fp)]

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
        else:
            # print('purged', box)
            pass

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

    for fp in annotation_fps:
        annotation_fp = os.path.join(root_dir, 'annotations', fp)
        bboxes = get_bboxes(annotation_fp)

        areas = [(box[3] - box[1]) * (box[2] - box[0]) for box in bboxes]

        return {'avg_area' : np.mean(areas), 'min_area' : np.min(areas), 'max_area' : np.max(areas)}

#TESTS:
if __name__ == '__main__':
    #TESTING purge_invalid_bboxes:
    dummy_boxes = [[0, 5, 0, 6], [1, 3, 2, 5], [44.0, 224.0, 48.0, 224.0]]
    # print(purge_invalid_bboxes(dummy_boxes))

    #TESTING bbox_dataset_statistics:
    import json

    config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
    DATA_FP = config['data_filepath_local']

    print(bbox_dataset_statistics(DATA_FP))
