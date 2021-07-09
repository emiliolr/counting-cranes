import os
import sys
import shutil
import json
import argparse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.getcwd(), 'density_estimation', 'ASPDNet'))
sys.path.append(os.path.join(os.getcwd(), 'object_detection'))

from utils import *
from density_estimation.ASPDNet_model import ASPDNetLightning
from density_estimation.ASPDNet.model import ASPDNet
from object_detection.faster_rcnn_model import FasterRCNNLightning

def run_pipeline(mosaic_fp, tile_save_dir, model_name, model_save_fp):

    """
    A master function that assembles all pipeline elements.
    This function predicts a total count for a given mosaic (i.e., a flight line).
    Inputs:
     - mosaic_fp: the filepath to the mosaic to predict a total count for
     - tile_save_dir: the directory to save the mosaic tiles to
     - model_name: the model to use for the prediction component... currently, should be one of faster_rcnn or ASPDNet
     - model_save_fp: the saved model as either a .pth or .ckpt file
    Outputs:
     - A master count for the input image set or mosaic (also saves run results)
    """

    #TILE MOSAIC:
    #  - read in mosaic
    #  - tile according to protocol in "bird_dataset.py"
    #    - this includes pre-processing... no GT though!
    #  - save tiles (?)
    #    - seems like this will probably be necessary to avoid OOM errors
    mosaic = Image.open(mosaic_fp).convert('RGB')
    mosaic_tiles = tiling_w_o_overlap_NO_BBOXES(mosaic, tile_size = (200, 200))

    if os.path.isdir(tile_save_dir): #if the save directory exists, it will be removed
        shutil.rmtree(tile_save_dir)
    os.mkdir(tile_save_dir) #create an empty directory

    for i, tile in enumerate(mosaic_tiles): #saving the tiles to the created directory
        tile = Image.fromarray(tile)
        tile.save(os.path.join(tile_save_dir, f'tile_{i}.tif'))

    #PREDICT ON TILES:
    #  - connect tiles to PyTorch dataloader
    #    - do pre-processing here (divide by 255, normalization, to PyTorch tensor)
    #  - instantiate the pytorch_lightning model and read in the state dict of the trained model provided
    #  - loop through the tiles and predict on each (keep running count)
    tile_dataset = BirdDatasetPREDICTION(tile_save_dir)
    tile_dataloader = DataLoader(tile_dataset, batch_size = 32, shuffle = False, num_workers = 4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #  loading the model from either a PyTorch Lightning checkpoint or a PyTorch model save
    #    TODO: make sure to pass necessary hyperparams here... ge them out of a config file!
    if model_name == 'faster_rcnn':
        if model_save_fp.endswith('.pth'):
            model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2)
            model.load_state_dict(model_save_fp)
            pl_model = FasterRCNNLightning(model)
        elif model_save_fp.endswith('.ckpt'):
            model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2)
            pl_model = FasterRCNNLightning.load_from_checkpoint(model_save_fp, model = model)
        else:
            raise NameError('File is not of type .pth or .ckpt')
    elif model_name == 'ASPDNet':
        if model_save_fp.endswith('.pth'):
            model = ASPDNet(allow_neg_densities = False)
            model.load_state_dict(model_save_fp)
            pl_model = ASPDNetLightning(model, lr = 1e-7)
        elif model_save_fp.endswith('.ckpt'):
            model = ASPDNet(allow_neg_densities = False)
            pl_model = ASPDNetLightning.load_from_checkpoint(model_save_fp, model = model, lr = 1e-7)
        else:
            raise NameError('File is not of type .pth or .ckpt')
    else:
        raise NameError(f'Model "{model_name}" is not a supported model type')

    total_count = 0
    for tile_batch in tile_dataloader:
        if model_name == 'faster_rcnn':
            tile_batch = list(tile_batch) #turning it into a list of tensors, as required by Faster R-CNN in PyTorch

        tile_counts = pl_model.predict_counts(tile_batch) #predicting on the tiles and extracting counts
        total_count += sum(tile_counts) #adding in the counts for this batch of tiles

    #CLEANUP:
    #  - save and return run results

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

#TODO: test out pipeline more to make sure everything's happening correctly!
#  - comment out stuff progressively
if __name__ == '__main__':
    #ARGPARSER: use to collect run info from command line, then pass along to function
    #  - mosaic FP (required)
    #  - model... either faster_rcnn or ASPDNet (required)
    #  - model save FP... as a .ckpt or .pth (required)
    #    - assume that this will be ASPDNet or Faster R-CNN
    #  - tiling method... w/ or w/o overlap (required)
    #  - config filepath (required)
    #    - get model hyperparams from here!
    #  - CSV file to write results to (required)
    #    - create if doesn't exist, add to otherwise ("MOSAIC_FP, runtime, final count")
    parser = argparse.ArgumentParser()
    parser.add_argument('mosaic_fp', help = 'file path for mosaic')
    parser.add_argument('tile_save_dir', help = 'directory to save mosaic tiles to')
    parser.add_argument('model_fp', help = 'file path for model save; .ckpt or .pth')
    parser.add_argument('config_fp', help = 'file path for config file, containing hyperparameters')
    parser.add_argument('write_results_fp', help = 'file path to write pipeline run results to')
    # parser.add_argument('tiling_method', help = 'method for tiling mosaic; w_overlap or w_o_overlap')

    args = parser.parse_args()

    mosaic_fp = '/Users/emiliolr/Desktop/counting-cranes/image_stitching/stitched_parent_image_TEST.png'
    model = 'faster_rcnn'
    model_fp = 'initial_faster_rcnn.pth'
    # run_pipeline(mosaic_fp, 'mosaic_tiles')
