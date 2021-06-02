import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np

#TODO:
#  - add density map generation method (see saved links!)
#  - add image tiling (will have to be aware of annotations and annotation type!)

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

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        xmin = np.clip(int(float(boxes.find('bndbox/xmin').text)), 0, 1280)
        ymin = np.clip(int(float(boxes.find('bndbox/ymin').text)), 0, 720)
        xmax = np.clip(int(float(boxes.find('bndbox/xmax').text)), 0, 1280)
        ymax = np.clip(int(float(boxes.find('bndbox/ymax').text)), 0, 720)

        list_with_all_boxes.append([xmin, ymin, xmax, ymax])

    return list_with_all_boxes

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

def visualize_bboxes(image_fp, xml_fp):

    """
    Draws bboxes on image.
    Inputs:
      - image_fp: the image filepath
      - xml_fp: the xml filepath
    Outputs:
      - A PIL image w/bboxes drawn on
    """

    img = Image.open(image_fp)
    bboxes = get_bboxes(xml_fp)

    draw = ImageDraw.Draw(img)
    for b in bboxes:
        draw.rectangle(b, outline = 'red', width = 1)

    return img

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
