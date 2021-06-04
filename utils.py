import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from torch import as_tensor

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

def purge_invalid_bboxes(list_of_bboxes, return_tensor = False):

    """
    A convenience function to remove any invalid bounding boxes.
    Inputs:
     - list_of_bboxes: a list of lists that contains bboxes in [xmin, ymin, xmax, ymax] format
     - return_tensor: should we return a tensor?
    Outputs:
      - A final list (or tensor) of valid bboxes
    """

    final_bboxes = []
    for box in list_of_bboxes:
        xmin, ymin, xmax, ymax = box

        if xmin >= xmax or ymin >= ymax:
            continue
        final_bboxes.append(box)

    if return_tensor:
        return torch.as_tensor(final_bboxes)
    return final_bboxes

#TESTS:
if __name__ == '__main__':
    #TESTING purge_invalid_bboxes:
    dummy_boxes = [[0, 5, 0, 6], [1, 3, 2, 5]]
    print(purge_invalid_bboxes(dummy_boxes))
