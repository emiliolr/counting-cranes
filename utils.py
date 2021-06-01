import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def get_bboxes(xml_fp):

    """
    Reads in all bounding boxes for an image.
    XML file is assumed to be in PascalVOC format.
    From https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python.
    Inputs:
      - xml_fp: an xml file w/bbox annotations
    Outputs:
      - a list of lists w/all bbox annotations in (xmin, ymin, xmax, ymax) format
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text

        xmin = int(float(boxes.find("bndbox/xmin").text)) #have to do the float thing for Kyle's CVAT annotations
        ymin = int(float(boxes.find("bndbox/ymin").text))
        xmax = int(float(boxes.find("bndbox/xmax").text))
        ymax = int(float(boxes.find("bndbox/ymax").text))

        list_with_all_boxes.append([xmin, ymin, xmax, ymax])

    return list_with_all_boxes

def visualize_bboxes(image_fp, xml_fp):

    """
    Draws bboxes on image.
    Inputs:
      - image_fp: the image filepath
      - xml_fp: the xml filepath
    Outputs:
      - a PIL image w/bboxes drawn on
    """

    img = Image.open(image_fp)
    bboxes = get_bboxes(xml_fp)

    draw = ImageDraw.Draw(img)
    for b in bboxes:
        draw.rectangle(b, outline = 'red', width = 1)
    return img
