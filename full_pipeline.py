import os
import sys
import shutil
import json
import csv
import time
import argparse
from datetime import date
import gc
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.getcwd(), 'density_estimation', 'ASPDNet'))
sys.path.append(os.path.join(os.getcwd(), 'object_detection'))

from utils import *
from density_estimation.ASPDNet_model import ASPDNetLightning
from density_estimation.ASPDNet.model import ASPDNet
from object_detection.faster_rcnn_model import *

def run_pipeline(mosaic_fp, model_name, model_save_fp, write_results_fp, num_workers, model_hyperparams = None, save_preds = False):

    """
    A wrapper function that assembles all pipeline elements.
    This function predicts a total count for a given mosaic (i.e., a flight line).
    Inputs:
     - mosaic_fp: the filepath to the mosaic to predict a total count for
     - model_name: the model to use for the prediction component... currently, should be one of faster_rcnn or ASPDNet
     - model_save_fp: the saved model as either a .pth or .ckpt file
     - write_results_fp: the CSV file to write the run results to
     - num_workers: the number of workers to use for the tile dataloader
     - model_hyperparams: any hyperparameters to use for the model
    Outputs:
     - A total count for the input mosaic (also saves run results to desired CSV file)
    """

    start_time = time.time()

    #TILE MOSAIC:
    mosaic = Image.open(mosaic_fp).convert('RGB')
    print(f'Tiling mosaic of size {mosaic.size[0]}x{mosaic.size[1]}...')
    tile_size = (200, 200)
    mosaic_tiles = tiling_w_o_overlap_NO_BBOXES(mosaic, tile_size = tile_size) #tile the mosaic into non-overlapping tiles

    if os.path.isdir('mosaic_tiles'): #if the tile save directory exists, it will be removed
        shutil.rmtree('mosaic_tiles')
    os.mkdir('mosaic_tiles') #create an empty directory

    for i, tile in enumerate(mosaic_tiles): #saving the tiles to the created directory
        tile = Image.fromarray(tile)
        tile.save(os.path.join('mosaic_tiles', f'tile_{i}.tif'))
    print('Done tiling mosaic!')

    #  freeing up memory by collecting the mosaic stuff that is no longer needed (it was saved)
    del mosaic
    del mosaic_tiles
    gc.collect()

    #PREDICT ON TILES:
    tile_dataset = BirdDatasetPREDICTION('mosaic_tiles', model_name)
    tile_dataloader = DataLoader(tile_dataset, batch_size = 32, shuffle = False, collate_fn = collate_tiles_PREDICTION, num_workers = num_workers)
    print(f'\nPredicting on {len(tile_dataset)} tiles...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #  grabbing any constructor hyperparams - currently, only necessary for our Faster R-CNN impelementation!
    if model_hyperparams is not None:
        constructor_hyperparams = model_hyperparams['constructor_hyperparams']

    #  loading the model from either a PyTorch Lightning checkpoint or a PyTorch model save
    print(f'\tLoading the saved {model_name} model...')
    if model_name == 'faster_rcnn':
        if model_save_fp.endswith('.pth'):
            model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2, **constructor_hyperparams).to(device) #making sure to pass in the constructor hyperparams here
            model.load_state_dict(torch.load(model_save_fp))
            pl_model = FasterRCNNLightning(model)
        elif model_save_fp.endswith('.ckpt'):
            model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2, **constructor_hyperparams).to(device)
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

    print('\tProducing counts...')

    pred_start_time = time.time()

    total_count = 0
    pl_model.model.eval() #making sure we're in eval mode...
    for i, batch in enumerate(tile_dataloader):
        print(f'\t\tBatch {i + 1}/{len(tile_dataloader)}')
        tile_batch, tile_nums = batch #getting out the content from the dataloader
        tile_batch = tile_batch.to(device) #loading the batch onto the same device as the model

        if model_name == 'faster_rcnn':
            tile_batch = list(tile_batch) #turning it into a list of tensors, as required by Faster R-CNN

        with torch.no_grad(): #disabling gradient calculations... not necessary, since we're just doing forward passes!
            tile_preds = pl_model(tile_batch) #getting predictions... not yet counts!

        if model_name == 'faster_rcnn': #predicting on the tiles and extracting counts for each tile
            tile_counts = [len(p['boxes'].tolist()) for p in tile_preds]
            total_count += sum(tile_counts) #adding in the counts for this batch of tiles
        elif model_name == 'ASPDNet':
            total_count += float(tile_preds.sum()) #adding in the counts

        #Saving predictions as we go
        if save_preds:
            os.mkdir('mosaic_tiles/predictions') #create an empty directory for preds
            if model_name == 'faster_rcnn': #saving tiles w/bboxes overlaid
                for i, (img, num) in enumerate(zip(tile_batch, tile_nums)):
                    img = (np.moveaxis(img.numpy(), 0, -1) * 255).astype(np.uint8)
                    pred_boxes = tile_preds[i]['boxes'].tolist()

                    pil_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    for b in pred_boxes: #drawing bboxes onto the tile
                        draw.rectangle(b, outline = 'red', width = 1)
                    pil_img.save(os.path.join('mosaic_tiles', 'predictions', f'pred_tile_{num}.tif'))
            elif model_name == 'ASPDNet': #saving the pred densities for each tile
                cm = plt.get_cmap('jet')
                for den, num in zip(list(tile_preds), tile_nums):
                    colored_image = cm(den.numpy()) #applying the color map... makes it easier to look at!

                    pil_img = Image.fromarray((colored_image * 255).astype(np.uint8)[ : , : , : 3]) #converting to PIL image
                    pil_img.save(os.path.join('mosaic_tiles', 'predictions', f'pred_tile_{num}.tif'))

    pred_time = time.time() - pred_start_time

    print('Done with prediction!')
    if save_preds:
        print(f'Predictions saved at {os.path.join("mosaic_tiles", "predictions")}')

    #SAVING/RETURNING RESULTS:
    fields = ['date', 'time', 'mosaic_fp', 'num_tiles', 'total_count', 'model', 'total_run_time', 'prediction_run_time']
    curr_time = str(time.strftime('%H:%M:%S', time.localtime()))
    curr_date = str(date.today())
    pipeline_time = time.time() - start_time
    new_row = [curr_date, curr_time, mosaic_fp, len(tile_dataset), int(total_count), model_name, pipeline_time, pred_time] #all of the run results to include

    if not os.path.isfile(write_results_fp): #either creating a new results CSV or adding to the existing file
        with open(write_results_fp, 'w') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(fields)
            csvwriter.writerow(new_row)
    else:
        with open(write_results_fp, 'a') as file:
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
     - model_name: either Faster R-CNN or ASPDNet
    """

    def __init__(self, root_dir, model_name):
        self.root_dir = root_dir
        self.tile_fps = sorted(os.listdir(self.root_dir))

        self.transforms = []
        if model_name == 'ASPDNet': #PyTorch's Faster R-CNN impelementation handles normalization...
            self.transforms.append(A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 1))
        self.transforms.append(ToTensorV2())
        self.transforms = A.Compose(self.transforms)

    def __getitem__(self, index):
        tile_fp = os.path.join(self.root_dir, self.tile_fps[index])
        tile_num = int(tile_fp.split('_')[-1].replace('.tif', '')) #grabbing this for saving preds
        tile = Image.open(tile_fp).convert('RGB')
        tile = np.array(tile)

        preprocessed_tile = tile / 255
        preprocessed_tile = self.transforms(image = preprocessed_tile)['image'].float() #making sure that its dtype is float32

        return preprocessed_tile, tile_num

    def __len__(self):
        return len(self.tile_fps)

def collate_tiles_PREDICTION(batch):
    """
    A workaround to ensure that we can retrieve the tile number for saving pipeline predictions.
    Inputs:
      - batch: a list of tuples w/format [(tile, tile_num), ...]
    Outputs:
      - A tuple w/a list of tiles and a list of tile numbers
    """
    tiles = torch.stack([b[0] for b in batch])
    tile_nums = [b[1] for b in batch]

    return tiles, tile_nums

def str2bool(arg):
    """
    A simple workaround to ensure that bools from the argument parser are handled correctly.
    From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse.
    Inputs:
      - arg: the string argument passed to the parser
    Outputs:
      - True or False, depending on the inputted string
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #an argument parser to collect arguments from the user

    #  required args
    parser.add_argument('mosaic_fp', help = 'file path for mosaic')
    parser.add_argument('model_name', help = 'the model name; either ASPDNet or faster_rcnn')
    parser.add_argument('model_fp', help = 'file path for model save; .ckpt or .pth')
    parser.add_argument('write_results_fp', help = 'file path to write pipeline run results to')

    #  optional args
    parser.add_argument('-nw', '--num_workers', help = 'the number of workers to use in the tile dataloader', type = int, default = 0)
    parser.add_argument('-cfp', '--config_fp', help = 'file path for config, containing hyperparameters for model', default = None)
    parser.add_argument('-sp', '--save_preds', help = 'save predictions for tiles?', type = str2bool, default = False)

    args = parser.parse_args()

    if args.config_fp is not None: #getting the config file
        config = json.load(open(args.config_fp, 'r'))
        model_hyperparams = config[args.model_name + '_params'] #see config.json for structure of config file
    else:
        model_hyperparams = None

    run_pipeline(args.mosaic_fp, args.model_name, args.model_fp, args.write_results_fp, args.num_workers, model_hyperparams = model_hyperparams, save_preds = args.save_preds)
