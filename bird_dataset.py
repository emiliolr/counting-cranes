import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from utils import get_bboxes

import os
from PIL import Image

#TODO: we will have to do some special things to hold out 2018 data for evaluation of overlap
#TODO: figure out where image tiling fits in... maybe in constructor?
class BirdDataset(Dataset):

    """
    A custom PyTorch dataset for our bird imagery.
    """

    def __init__(self, root_dir, transforms, annotation_mode = 'bboxes'):
        self.root_dir = root_dir
        self.transforms = transforms

        #Load the images (tif) and annotations (xml) - sorting to make sure they're in the same order!
        self.image_fps = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
        self.annotation_fps = sorted(os.listdir(os.path.join(self.root_dir, 'annotations')))

        #Checking that there's an annotation for each image
        imgs = [i.split('.')[0] for i in self.image_fps] #getting the filenames
        annots = [a.split('.')[0] for a in self.annotation_fps]
        assert imgs == annots, 'Annotation and image filepaths don\'t align'

        #Checking that the annotation type is valid
        assert annotation_mode in ['bboxes', 'points', 'regression'], 'Choose one of bboxes, point, or regression for annotation type'

    def __getitem__(self, index):
        #Get the particular image and annotated bboxes
        image_fp = os.path.join(self.root_dir, 'images', self.image_fps[index])
        annotation_fp = os.path.join(self.root_dir, 'annotations', self.annotation_fps[index])

        #Pulling in the image and bboxes
        image = Image.open(image_fp).convert('L') #TODO: if we want three repeated bands (shape requirements?), convert to RGB!
        bboxes = get_bboxes(annotation_fp)

        #Turning the bboxes into tensors
        bboxes = torch.as_tensor(bboxes, dtype = torch.float32)

        #Produce labels for bboxes - we aren't performing classification, just trying to identify birds
        labels = torch.ones((len(bboxes), ))

        #Performing transformations on input image
        if self.transforms is not None:
            img = self.transforms(img)

        #Putting together the target dictionary, as specified in Faster R-CNN PyTorch docs
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels

        return image, target

    def __len__(self):
        return len(self.image_fps)

#TODO: see the Faster R-CNN preprocessing regimen packaged in PyTorch!
def get_transforms(train = True):

    """
    A convenience function to hold transformations for train/test.
    Inputs:
      - train: should we return the training or testing transforms?
    Outputs:
      - Composed PyTorch transformation chain
    """

    pass
