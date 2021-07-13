import os
import sys
import shutil
import json
import csv
import time
import argparse
from datetime import date
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

#TODO: this might not work for Colab!
sys.path.append(os.path.join(os.getcwd(), 'density_estimation', 'ASPDNet'))
sys.path.append(os.path.join(os.getcwd(), 'object_detection'))

from utils import *
from density_estimation.ASPDNet_model import ASPDNetLightning
from density_estimation.ASPDNet.model import ASPDNet
from object_detection.faster_rcnn_model import *

def run_pipeline(mosaic_fp, tile_save_dir, model_name, model_save_fp):

    """
    A wrapper function that assembles all pipeline elements.
    This function predicts a total count for a given mosaic (i.e., a flight line).
    Inputs:
     - mosaic_fp: the filepath to the mosaic to predict a total count for
     - tile_save_dir: the directory to save the mosaic tiles to
     - model_name: the model to use for the prediction component... currently, should be one of faster_rcnn or ASPDNet
     - model_hyperparams: model hyperparameters to use
     - model_save_fp: the saved model as either a .pth or .ckpt file
    Outputs:
     - A master count for the input image set or mosaic (also saves run results)
    """

    #TILE MOSAIC:
    mosaic = Image.open(mosaic_fp).convert('RGB')
    print(f'Tiling mosaic of size {mosaic.size[0]}x{mosaic.size[1]}...')
    tile_size = (200, 200)
    mosaic_tiles = tiling_w_o_overlap_NO_BBOXES(mosaic, tile_size = tile_size) #tile the mosaic into non-overlapping tiles

    if os.path.isdir(tile_save_dir): #if the save directory exists, it will be removed
        shutil.rmtree(tile_save_dir)
    os.mkdir(tile_save_dir) #create an empty directory

    for i, tile in enumerate(mosaic_tiles): #saving the tiles to the created directory
        tile = Image.fromarray(tile)
        tile.save(os.path.join(tile_save_dir, f'tile_{i}.tif'))
    print('Done tiling mosaic!')

    #PREDICT ON TILES:
    tile_dataset = BirdDatasetPREDICTION(tile_save_dir)
    tile_dataloader = DataLoader(tile_dataset, batch_size = 32, shuffle = False) #TODO: add more workers here??
    print(f'\nPredicting on {len(tile_dataset)} tiles...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #  loading the model from either a PyTorch Lightning checkpoint or a PyTorch model save
    print(f'\tLoading the saved {model_name} model...')
    if model_name == 'faster_rcnn':
        if model_save_fp.endswith('.pth'):
            model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2).to(device)
            model.load_state_dict(torch.load(model_save_fp))
            pl_model = FasterRCNNLightning(model)
        elif model_save_fp.endswith('.ckpt'):
            model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2).to(device)
            pl_model = FasterRCNNLightning.load_from_checkpoint(model_save_fp, model = model)
        else:
            raise NameError('File is not of type .pth or .ckpt')
    elif model_name == 'ASPDNet':
        if model_save_fp.endswith('.pth'):
            model = ASPDNet(allow_neg_densities = False).to(device)
            model.load_state_dict(torch.load(model_save_fp))
            pl_model = ASPDNetLightning(model)
        elif model_save_fp.endswith('.ckpt'):
            model = ASPDNet(allow_neg_densities = False).to(device)
            pl_model = ASPDNetLightning.load_from_checkpoint(model_save_fp, model = model)
        else:
            raise NameError('File is not of type .pth or .ckpt')
    else:
        raise NameError(f'Model "{model_name}" is not a supported model type')

    #TODO: you'll want to come through here to make sure that these predictions are looking good qualitatively (are trained models coming in correctly?)
    #  - maybe save to a global variable and visualize them just to make sure things are correct!
    print('\tProducing counts...')
    total_count = 0
    for i, tile_batch in enumerate(tile_dataloader):
        print(f'\t\tBatch {i + 1}/{len(tile_dataloader) + 1}')
        if model_name == 'faster_rcnn':
            tile_batch = list(tile_batch) #turning it into a list of tensors, as required by Faster R-CNN

        tile_counts = pl_model.predict_counts(tile_batch) #predicting on the tiles and extracting counts for each tile
        total_count += sum(tile_counts) #adding in the counts for this batch of tiles
        break #TODO: remove this so that we can produce a full dataset count!

    print('Done with prediction!')

    #SAVING/RETURNING RESULTS:
    fields = ['date', 'time', 'mosaic_fp', 'num_tiles', 'total_count', 'model']
    curr_time = str(time.strftime('%H:%M:%S', time.localtime()))
    curr_date = str(date.today())
    new_row = [curr_date, curr_time, mosaic_fp, len(tile_dataset), total_count, model_name] #all of the run results to include

    if not os.path.isfile('pipeline_results.csv'): #either creating a new results CSV or adding to the existing file
        with open('pipeline_results.csv', 'w') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(fields)
            csvwriter.writerow(new_row)
    else:
        with open('pipeline_results.csv', 'a') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(new_row)
    print('\nResults saved!')

    return total_count

def tiling_w_o_overlap_NO_BBOXES(image, tile_size = (224, 224)):

    """
    A basic version of tiling w/o overlap that doesn't accept annotations.
    Inputs:
     - image: the image to tile w/o overlap
     - tile_size: the size of the tile, in format (width, height)
    Outputs:
     - A list of tiles as numpy arrays
    """

    padded_image = pad_parent_for_tiles(image, tile_size)
    tile_width, tile_height = tile_size
    image_width, image_height = padded_image.size

    tiles = []
    for h in range(0, (image_height + 1) - tile_height, tile_height):
        for w in range(0, (image_width + 1) - tile_width, tile_width):
            coords = (w, h, w + tile_width, h + tile_height)
            crop = A.Crop(*coords)
            t = crop(image = np.array(padded_image))

            tiles.append(t['image'])

    return tiles

class BirdDatasetPREDICTION(Dataset):

    """
    A reduced version of BirdDataset to help read in and preprocess mosaic tiles for prediction.
    Inputs:
     - root_dir: the root directory for mosaic tiles
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tile_fps = sorted(os.listdir(self.root_dir))
        self.transforms = A.Compose([A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 1),
                                     ToTensorV2()])

    def __getitem__(self, index):
        tile_fp = os.path.join(self.root_dir, self.tile_fps[index])
        tile = Image.open(tile_fp).convert('RGB')
        tile = np.array(tile)

        preprocessed_tile = tile / 255
        preprocessed_tile = self.transforms(image = preprocessed_tile)['image']

        return preprocessed_tile

    def __len__(self):
        return len(self.tile_fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #an argument parser to collect arguments from the user

    parser.add_argument('mosaic_fp', help = 'file path for mosaic')
    parser.add_argument('tile_save_dir', help = 'directory to save mosaic tiles to')
    parser.add_argument('model_name', help = 'the model name; either ASPDNet or faster_rcnn')
    parser.add_argument('model_fp', help = 'file path for model save; .ckpt or .pth')
    parser.add_argument('write_results_fp', help = 'file path to write pipeline run results to')

    args = parser.parse_args()

    run_pipeline(args.mosaic_fp, args.tile_save_dir, args.model_name, args.model_fp)
