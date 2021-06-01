import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from utils import get_bboxes

import os

class BirdDataset(Dataset):

    """
    A custom PyTorch dataset for our bird imagery.
    NOTE: we will have to do some special things to hold out 2018 data for evaluation of overlap.
    """

    def __init__(self, root_dir, transforms):
        self.root_fp = root_dir
        self.transforms = transforms

        #Load the images (tif) and annotations (xml) - sorting to make sure they're in the same order!
        self.image_fps = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.annotation_fps = sorted(os.listdir(os.path.join(root_dir, 'annotations')))

        #Checking that there's an annotation for each image
        imgs = [i.split('.')[0] for i in self.image_fps] #getting everything except the file type
        annots = [a.split('.')[0] for a in self.annotation_fps]
        assert imgs == annots, 'Annotation and image filepaths don\'t align'

    def __getitem__(self, index):
        #Get the particular image and annotated bboxes
        image_fp = os.path.join(root_dir, 'images', self.image_fps[index])
        annotation_fp = os.path.join(root_dir, 'annotations', self.annotation_fps[index])

        #Pulling in the image and bboxes
        image = Image.open(image_fp)
        bboxes = get_bboxes(annotation_fp)

        #Turning the bboxes into tensors
        bboxes = torch.as_tensor(bboxes, dtype = torch.float32)

        #Performing transformations on input image
        #  TODO: add image tiling here
        if self.transforms is not None:
            img = self.transforms(img)

    def __len__(self):
        return len(self.img_fps)

def get_transforms(train = True):
    pass
