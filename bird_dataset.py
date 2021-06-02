import torch
from torch.utils.data import Dataset

from utils import *

import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

#TODO: we will have to do some special things to hold out 2018 data for evaluation of overlap
#TODO: figure out where image tiling fits in... maybe in constructor?
class BirdDataset(Dataset):

    """
    A custom PyTorch dataset for our bird imagery.
    Inputs:
      - root_dir: the root directory for imagery/annotations
      - transforms: a composed chain of transformations
      - annotation_mode: the type of annotations to use (bboxes, points, or regression)
      - num_tiles: the number of tiles to split each parent image into
    """

    def __init__(self, root_dir, transforms, annotation_mode = 'bboxes', num_tiles = 18):
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotation_mode = annotation_mode
        self.num_tiles = num_tiles

        #Load the images (tif) and annotations (xml) - sorting to make sure they're in the same order!
        self.image_fps = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
        self.annotation_fps = sorted(os.listdir(os.path.join(self.root_dir, 'annotations')))

        #Checking that there's an annotation for each image
        imgs = [i.split('.')[0] for i in self.image_fps] #getting the filenames
        annots = [a.split('.')[0] for a in self.annotation_fps]
        assert imgs == annots, 'Annotation and image filepaths don\'t align'

        #Checking that the annotation type is valid
        assert annotation_mode in ['bboxes', 'points', 'regression'], 'Choose one of bboxes, point, or regression for annotation type'

    #TODO: modify this behavior based on which type of annotations we want!
    def __getitem__(self, index):
        if self.annotation_mode == 'bboxes':
            #Get the particular image and annotated bboxes
            image_fp = os.path.join(self.root_dir, 'images', self.image_fps[index])
            annotation_fp = os.path.join(self.root_dir, 'annotations', self.annotation_fps[index])

            #Pulling in the image and bboxes
            image = Image.open(image_fp).convert('RGB')
            bboxes = get_bboxes(annotation_fp)

            #Produce labels for bboxes - we aren't performing classification, just trying to identify birds
            labels = np.ones((len(bboxes), ))

            #Performing augmentations on the parent image
            transformed = self.transforms(image = np.array(image))
            image = transformed['image']

            #Tiling the parent image
            #  TODO: mess w/this once you want to introduce more than random tiling
            tiled_images = []
            tiled_bboxes = []
            tiled_class_labels = []
            tile_method = get_tiling_method('random')
            for i in range(self.num_tiles):
                tile = tile_method(image = image, bboxes = bboxes, class_labels = labels)
                tiled_images.append(tile['image'])
                tiled_bboxes.append(tile['bboxes'])
                tiled_class_labels.append(tile['class_labels'])

            #Ensuring that the return is formatted correctly for Faster R-CNN
            batch_of_tiles = []
            for img, boxes, labels in zip(tiled_images, tiled_bboxes, tiled_class_labels):
                target_dict = {}
                target_dict['boxes'] = torch.as_tensor(boxes, dtype = torch.float32)
                target_dict['labels'] = torch.as_tensor(labels, dtype = torch.int64)
                batch_of_tiles.append((img, target_dict))

            return batch_of_tiles

    def __len__(self):
        return len(self.image_fps)

def get_transforms(train = True):

    """
    A convenience function to hold transformations for the train and test sets.
    NOTE: PyTorch's Faster R-CNN impelementation seems to handle normalization.
    Inputs:
      - train: should we return the training or testing transforms?
    Outputs:
      - Composed PyTorch transformation chain
    """

    transforms = []
    #TODO: add albumentation augmentations here!

    return A.Compose(transforms)

def collate_w_tiles(batch):

    """
    A workaround to ensure that we get the right output for each batch in the DataLoader.
    For simplicity, use a batch size of 1... one parent image becomes any sub-images!
    Inputs:
      - batch: a list of lists of tuples w/format [](image, target), ...]
    Outputs:
      - A tuple w/the list of images and list of target dictionaries
    """

    tiles = batch[0] #grabbing the only element of the batch
    images = [t[0] for t in tiles] #produces a list of tensors
    targets = [t[1] for t in tiles] #produces a list of dictionaries

    return images, targets

def get_tiling_method(type = 'random'):
    if type == 'random':
        tiling = A.Compose([A.RandomCrop(224, 244),
                            ToTensorV2()],
                            bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels']))
        return tiling
    elif type == 'overlap':
        pass
    elif type == 'no_overlap':
        pass
