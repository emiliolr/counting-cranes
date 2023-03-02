import time

import argparse
import os, sys
import gc
import shutil
from PIL import Image
import numpy as np

from utils import *
import torch
from datetime import date
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser() #an argument parser to collect arguments from the user
parser.add_argument('-tw', '--tile-width', type=int, default=200, help='width of the tiles')
parser.add_argument('-th', '--tile-height', type=int, default=200, help='height of the tiles')
parser.add_argument('-f', '--file', type=str, help='filepath for mosaic', default=None)
parser.add_argument('-d', '--directory', type=str, help='filepath for directory of mosaics', default=None)
#parser.add_argument('-p', '--padding', action='store_false', help='pad black pixels if tiles don\'t fit neatly (defaults to true)')         # padding 

tiles_dir = 'mosaic_tiles'

def tile_dir() -> str:
    return tiles_dir

def tiling_w_o_overlap_NO_BBOXES(image_file, tile_size = (200, 200)):

    """
    A basic version of tiling w/o overlap that doesn't accept annotations.
    Inputs:
     - image: the image to tile w/o overlap
     - tile_size: the size of the tile, in format (width, height)
    Outputs:
     - Nothing... tiles are saved as part of the process
    """
    img_name = image_file.split(os.path.sep)[-1].split('.')[0]
    image = Image.open(image_file)
    padded_image = pad_parent_for_tiles(image, tile_size)
    tile_width, tile_height = tile_size
    image_width, image_height = padded_image.size

    #  freeing up memory
    del image
    gc.collect()

    zero_tensor = torch.zeros(*(tile_size[0], tile_size[1], 3)) #represents a black tile

    i = 0
    for h in range(0, (image_height + 1) - tile_height, tile_height):
        for w in range(0, (image_width + 1) - tile_width, tile_width):
            coords = (w, h, w + tile_width, h + tile_height)
            crop = A.Crop(*coords)
            t = crop(image = np.array(padded_image))

            tile = Image.fromarray(t['image'])
            tensor_tile = torch.from_numpy(np.array(tile))
            if not bool(torch.all(torch.eq(tensor_tile, zero_tensor))): #only saving if it's not an all-black tile
                tile.save(os.path.join('mosaic_tiles', f'{img_name}_{i}.tif'))
                i += 1

def tile_file(file, tile_size = (200,200)):
    # make tile directory if it doesn't exist
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
    tw, th = tile_size
    img_name = file.split(os.path.sep)[-1].split('.')[0]
    # open image and store as numpy array
    img = Image.open(file).convert('L')
    img_array = np.asarray(img)

    # pad image to be divisible by tw and th
    padded_img_array = np.pad(img_array, ((0, int(img.width % tw)), (0, int(img.height % th))))
    
    # free up some memory
    del img, img_array
    gc.collect()

    tile_count = 0
    for row in range(0, padded_img_array.shape[0]-1, tw):
        for col in range(0, padded_img_array.shape[1]-1, th):
            tile_name = os.path.join(tiles_dir, f"{img_name}_tile_{tile_count}.tif")
            # if tile already exists, skip
            if not os.path.isfile(tile_name):
                tile_array = padded_img_array[row:row+tw][:,col:col+th]
                # if tile is all zeroes, skip
                if tile_array.any():
                    tile_img = Image.fromarray(tile_array, 'L')
                    tile_img.save(tile_name)
                    tile_count += 1

def tile_dir(dir, tile_size = (200, 200)):
    # make tile directory if it doesn't exist
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
    tw, th = tile_size
    for (root, _, files) in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            print(path)

            # Below is copied from tile_file

            img_name = path.split(os.path.sep)[-1].split('.')[0]
            # open image and store as numpy array
            img = Image.open(path).convert('L')
            img_array = np.asarray(img)

            # pad image to be divisible by tw and th
            padded_img_array = np.pad(img_array, ((0, int(img.width % tw)), (0, int(img.height % th))))
            
            # free up some memory
            del img, img_array
            gc.collect()

            tile_count = 0
            for row in range(0, padded_img_array.shape[0]-1, tw):
                for col in range(0, padded_img_array.shape[1]-1, th):
                    tile_name = os.path.join(tiles_dir, f"{img_name}_tile_{tile_count}.tif")
                    # if tile already exists, skip
                    if not os.path.isfile(tile_name):
                        tile_array = padded_img_array[row:row+tw][:,col:col+th]
                        # if tile is all zeroes, skip
                        if tile_array.any():
                            tile_img = Image.fromarray(tile_array, 'L')
                            tile_img.save(tile_name)
                            tile_count += 1




if __name__ == '__main__':
    args = parser.parse_args()
    
    # if both or neither file or directory were passed
    if bool(args.file == None) == bool(args.directory == None):
        raise argparse.ArgumentTypeError('Include either \'--file\' or \'--directory\' in command line')
        sys.exit(1)

    tw, th = args.tile_width, args.tile_height
    # tile a single .tif file
    if args.file and (args.file.endswith(".tif") or args.file.endswith(".TIF")):
        tile_file(args.file, (tw, th))
    # tile all the .tif files in a directory
    elif args.directory:
        tile_dir(args.directory, (tw, th))

    # Testing runtime of different functions
    """ new_times = []
    old_times = []
    N = 100
    for i in range(N):
        start = time.time()
        new_tiling_function(args.file, tw, th)
        new_times.append(time.time()-start)
        start = time.time()
        tiling_w_o_overlap_NO_BBOXES(Image.open(args.file))
        old_times.append(time.time()-start)

    print("Mean execution time for new function:", np.mean(np.array(new_times)))
    print("Mean execution time for old function:", np.mean(np.array(old_times))) """

