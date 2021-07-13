import torch
from torch.utils.data import Dataset

from utils import *
from density_estimation.generate_density import *

import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class BirdDataset(Dataset):

    """
    A custom PyTorch dataset for our bird imagery.
    Inputs:
      - root_dir: the root directory for imagery/annotations
      - transforms: a composed chain of transformations
      - annotation_mode: the type of annotations to use (bboxes, points, or regression)
      - tiling_method: the method to use for parent image tiling (random, w_o_overlap, w_overlap)
      - tile_size: the size of the tiles, in format (width, height)
      - num_tiles: the number of tiles to split each parent image into (only for random tiling)
      - max_neg_examples: the maximum number of negative examples (zero birds) to include in a training batch (only for random tiling)
      - sigma: the sigma to use for density map generation (only for point annotation)
    """

    def __init__(self, root_dir, transforms, annotation_mode = 'bboxes', tiling_method = 'w_o_overlap', tile_size = (224, 224), num_tiles = 18, max_neg_examples = 6, sigma = 1.5):
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotation_mode = annotation_mode
        self.tiling_method = tiling_method
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.max_neg_examples = max_neg_examples
        self.sigma = sigma

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
        image = Image.open(image_fp).convert('RGB')
        bboxes = get_bboxes(annotation_fp)

        #Produce labels for bboxes - we aren't performing classification, just trying to identify birds
        labels = np.ones((len(bboxes), ))

        #Performing augmentations on the parent image
        transformed = self.transforms(image = np.array(image), bboxes = bboxes, class_labels = labels)
        image = Image.fromarray(transformed['image'])
        bboxes = transformed['bboxes']
        labels = transformed['class_labels']

        #Tiling up the parent image using the desired tiling method
        if self.tiling_method == 'random':
            tiles, targets = random_tiling(image, bboxes, labels, self.num_tiles, self.max_neg_examples, self.tile_size, normalize = True if self.annotation_mode == 'points' else False)
        elif self.tiling_method == 'w_o_overlap':
            tiles, targets = tiling_w_o_overlap(image, bboxes, labels, self.tile_size, normalize = True if self.annotation_mode == 'points' else False)
        elif self.tiling_method == 'w_overlap':
            pass

        if self.annotation_mode == 'bboxes': #ensuring that the return is formatted correctly for Faster R-CNN
            batch_of_tiles = []

            img_name = self.image_fps[index].replace('.TIF', '').replace('.tif', '') #this is necessary for calculating metrics...
            for i, content in enumerate(zip(tiles, targets)):
                img, target = content
                img = img.float() #making it float32 rather than float64
                target_dict = {}
                target_dict['boxes'] = torch.as_tensor(target['boxes'], dtype = torch.float32)
                target_dict['labels'] = torch.as_tensor(target['labels'], dtype = torch.int64)
                batch_of_tiles.append((img, target_dict, f'{img_name}_{i}', f'{img_name}_{i}'))
        elif self.annotation_mode == 'regression':
            batch_of_tiles = []

            for content in zip(tiles, targets):
                img, target = content
                count = get_regression(target['boxes']) #turning bbox annotations into an integer count

                batch_of_tiles.append((img, count))
        elif self.annotation_mode == 'points':
            batch_of_tiles = []

            for content in zip(tiles, targets):
                img, target = content
                density = density_from_bboxes(target['boxes'], img, filter_type = 'fixed', sigma = self.sigma)
                density = cv2.resize(density, (density.shape[0] // 8, density.shape[1] // 8), interpolation = cv2.INTER_CUBIC) * 64 #this is required to ensure that the GT matches the pred density in shape
                density = torch.nn.functional.relu(torch.as_tensor(density, dtype = torch.float32)) #bounding out the negative values in the GT density map
                count = get_regression(target['boxes'])

                batch_of_tiles.append((img, density, count))

        return batch_of_tiles

    def __len__(self):
        return len(self.image_fps)

def get_transforms(task = 'object_detection', train = True):

    """
    A convenience function to hold transformations for all tasks.
    NOTE: PyTorch's Faster R-CNN impelementation handles normalization.
    Inputs:
      - task: either object_detection or density_estimation
    Outputs:
      - A chain of composed albumentations transformations
    """

    transforms = []
    if task == 'object_detection' and train:
        transforms.append(A.RandomBrightnessContrast(p = 0.5))
    elif task == 'density_estimation' and train:
        transforms.append(A.RandomBrightnessContrast(p = 0.5))
        transforms.append(A.HorizontalFlip(p = 0.5))
        transforms.append(A.VerticalFlip(p = 0.5))

    return A.Compose(transforms, bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels'], min_visibility = 0.2))

def collate_tiles_object_detection(batch):

    """
    A workaround to ensure that we get the right output for each batch in the DataLoader.
    For simplicity, use a batch size of 1 --> one parent image becomes any sub-images!
    This version is for object detection!
    Inputs:
      - batch: a list of lists of tuples w/format [(image, target), ...]
    Outputs:
      - A tuple w/the list of images, list of target dictionaries, and list of names for images/annotations
    """

    assert len(batch) == 1, 'Use a batch size of 1'
    tiles = batch[0] #grabbing the only element of the batch
    images = [t[0] for t in tiles] #produces a list of tensors
    targets = [t[1] for t in tiles] #produces a list of dictionaries
    img_names = [t[2] for t in tiles]
    annot_names = [t[3] for t in tiles]

    return images, targets, img_names, annot_names

def collate_tiles_regression(batch):

    """
    A workaround to ensure that we get the right output for each batch in the DataLoader.
    For simplicity, use a batch size of 1 --> one parent image becomes any sub-images!
    This version is for object regression!
    Inputs:
      - batch: a list of lists of tuples w/format [(image, count), ...]
    Outputs:
      - A tuple w/the list of images and list of counts
    """

    assert len(batch) == 1, 'Use a batch size of 1'
    tiles = batch[0]
    images = [t[0] for t in tiles]
    counts = [t[1] for t in tiles]

    return images, counts

def collate_tiles_density(batch):

    """
    A workaround to ensure that we get the right output for each batch in the DataLoader.
    For simplicity, use a batch size of 1 --> one parent image becomes any sub-images!
    This version is for object density!
    Inputs:
      - batch: a list of lists of tuples w/format [(image, density), ...]
    Outputs:
      - A tuple w/the list of images and list of densities
    """

    assert len(batch) == 1, 'Use a batch size of 1'
    tiles = batch[0]
    images = torch.stack([t[0] for t in tiles])
    densities = torch.stack([t[1] for t in tiles])
    counts = [t[2] for t in tiles]

    return images, densities, counts

def tiling_w_o_overlap(parent_image, bboxes, labels, tile_size = (224, 224), normalize = False):

    """
    A function to perform tiling w/o overlap on parent images.
    Inputs:
      - parent_image: the parent image to be tiled
      - bboxes: a list of lists containing all bboxes in the parent image
      - labels: a list of the class labels for bboxes in the parent image
      - tile_size: the size of the tile, in format (width, height)
      - normalize: should we normalize the tiles?
    Outputs:
      - A tuple of tiled images and new target dictionaries (contains bboxes and labels)
    """

    padded_parent = pad_parent_for_tiles(parent_image, tile_size)
    tile_width, tile_height = tile_size
    image_width, image_height = padded_parent.size
    normalization = A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 1)

    tiles = []
    targets = []
    for h in range(0, (image_height + 1) - tile_height, tile_height):
        for w in range(0, (image_width + 1) - tile_width, tile_width):
            coords = (w, h, w + tile_width, h + tile_height)
            transform_list = [A.Crop(*coords)]
            transforms = A.Compose(transform_list, bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels'], min_visibility = 0.2))
            t = transforms(image = np.array(padded_parent), bboxes = bboxes, class_labels = labels)

            if len(t['bboxes']) == 0:
                new_bboxes = torch.empty((0, 4), dtype = torch.float32)
            else:
                new_bboxes = purge_invalid_bboxes(t['bboxes'])
            target_dict = {'boxes' : new_bboxes, 'labels' : np.ones((len(new_bboxes), ))}

            image = t['image'] / 255
            if normalize:
                image = normalization(image = image)['image']
            tiles.append(ToTensorV2()(image = image)['image'])
            targets.append(target_dict)

    return tiles, targets

def random_tiling(parent_image, bboxes, labels, num_tiles, max_neg_examples, tile_size = (224, 224), normalize = False):

    """
    A function to perform random tiling on parent images.
    Inputs:
      - parent_image: the parent image to be tiled
      - bboxes: a list of lists containing all bboxes in the parent image
      - labels: a list of the class labels for bboxes in the parent image
      - num_tiles: the desired number of random tiles
      - max_neg_examples: the maximum negative examples (no birds in the tile) to keep
      - tile_size: the size of the tile, in format (width, height)
      - normalize: should we normalize the tiles?
    Outputs:
      - A tuple of tiled images and new target dictionaries (contains bboxes and labels)
    """
    transform_list = [A.RandomCrop(*tile_size)]
    random_crop = A.Compose(transform_list, bbox_params = A.BboxParams(format = 'pascal_voc', label_fields = ['class_labels'], min_visibility = 0.2))
    normalization = A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 1) #max pixel val is 1 here b/c images will be rescaled before normalization!

    tiles = []
    targets = []
    negative_example_ct = 0
    while len(tiles) < num_tiles:
        t = random_crop(image = np.array(parent_image), bboxes = bboxes, class_labels = labels)

        if len(t['bboxes']) == 0:
            if negative_example_ct >= max_neg_examples: #if we have too many negative examples, don't add the current neg example...
                continue
            negative_example_ct += 1

            new_bboxes = torch.empty((0, 4), dtype = torch.float32) #for negative examples, i.e., no bbox in the tile
        else:
            new_bboxes = purge_invalid_bboxes(t['bboxes']) #making sure to get rid of any invalid bboxes here
        target_dict = {'boxes' : new_bboxes, 'labels' : np.ones((len(new_bboxes), ))}

        image = t['image'] / 255 #re-scaling into [0, 1] range
        if normalize:
            image = normalization(image = image)['image'] #normalization using ImageNet stats... not necessary for object detection
        tiles.append(ToTensorV2()(image = image)['image'])
        targets.append(target_dict)

    return tiles, targets

#TESTS:
if __name__ == '__main__':
    import json
    from torch.utils.data import DataLoader

    config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
    DATA_FP = config['data_filepath_local']

    #TESTING THE DATASET:
    bird_dataset = BirdDataset(root_dir = DATA_FP, transforms = get_transforms('density_estimation', False), tiling_method = 'w_o_overlap', annotation_mode = 'points')
    bird_dataloader = DataLoader(bird_dataset, batch_size = 1, shuffle = False, collate_fn = collate_tiles_density)

    images, targets, counts = next(iter(bird_dataloader))
    print(f'Actual count is {sum(counts)} while count after resizing density is {sum([int(t.sum()) for t in targets])}')
    print([int(t.sum()) for t in targets])
    print(counts)

    # num_degen = 0
    # for i, data in enumerate(bird_dataloader):
    #   print('On image', i)
    #   images, targets, _, _ = data
    #   for d in targets:
    #     for b in d['boxes'].tolist():
    #       xmin, ymin, xmax, ymax = b
    #       if xmin >= xmax or ymin >= ymax:
    #         print('Invalid bbox:', b)
    #         num_degen += 1
    # print(f'We have {num_degen} invalid bboxes')
