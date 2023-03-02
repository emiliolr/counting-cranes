import argparse
import os
import gc
from PIL import Image
import numpy as np

tiles_dir = 'mosaic_tiles'

# for use outside of this file
def get_tile_dir() -> str:
    return tiles_dir

def tile_file(file, tile_size = (200,200)):
    if not file.endswith('.tif') and file.endswith('.TIF'):
        raise ValueError(f'File {file} is not a .tif or .TIF file')
    # make tile directory if it doesn't exist
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
    tw, th = tile_size

    img_name = file.split(os.path.sep)[-1].split('.')[0]
    # open image and store as numpy array
    img = Image.open(path).convert('L')
    img_array = np.asarray(img)

    # pad image to be divisible by tw and th
    padded_array = np.pad(img_array, ((0, int(img.width % tw)), (0, int(img.height % th))))
    
    # free up some memory
    del img, img_array
    gc.collect()

    tile_count = 0
    for row in range(0, padded_array.shape[0]-1, tw):
        for col in range(0, padded_array.shape[1]-1, th):
            # file name of new tile
            tile_name = os.path.join(tiles_dir, f"{img_name}_tile_{tile_count}.tif")
            # if tile already exists, skip
            if not os.path.isfile(tile_name):
                # segment padded array into a tw by th tile
                tile_array = padded_array[row:row+tw][:,col:col+th]
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
            print(f"Tiling {file}...")
            img_name = file.split('.')[0]
            # Below is copied from tile_file

            # open image and store as numpy array
            img = Image.open(path).convert('L')
            img_array = np.asarray(img)

            # pad image to be divisible by tw and th
            padded_array = np.pad(img_array, ((0, int(img.width % tw)), (0, int(img.height % th))))
            
            # free up some memory
            del img, img_array
            gc.collect()

            tile_count = 0
            for row in range(0, padded_array.shape[0]-1, tw):
                for col in range(0, padded_array.shape[1]-1, th):
                    # file name of new tile
                    tile_name = os.path.join(tiles_dir, f"{img_name}_tile_{tile_count}.tif")
                    # if tile already exists, skip
                    if not os.path.isfile(tile_name):
                        # segment padded array into a tw by th tile
                        tile_array = padded_array[row:row+tw][:,col:col+th]
                        # if tile is all zeroes, skip
                        if tile_array.any():
                            tile_img = Image.fromarray(tile_array, 'L')
                            tile_img.save(tile_name)
                            tile_count += 1
            
    print(f"Done tiling directory {dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tw', '--tile-width', type=int, default=200, help='width of the tiles')
    parser.add_argument('-th', '--tile-height', type=int, default=200, help='height of the tiles')
    parser.add_argument('-f', '--file', type=str, help='filepath for mosaic', default=None)
    parser.add_argument('-d', '--directory', type=str, help='filepath for directory of mosaics', default=None)
    #parser.add_argument('-p', '--padding', action='store_false', help='pad black pixels if tiles don\'t fit neatly (defaults to true)')         # padding 
    args = parser.parse_args()
    
    # if both or neither file or directory were passed
    if bool(args.file == None) == bool(args.directory == None):
        raise argparse.ArgumentTypeError('Include either \'--file\' or \'--directory\' in command line')
        return

    tw, th = args.tile_width, args.tile_height
    # tile a single .tif file
    if args.file:
        tile_file(args.file, (tw, th))
    # tile all the .tif files in a directory
    elif args.directory:
        tile_dir(args.directory, (tw, th))

    # Run this to test:
    # For dir : time python3 tile_mosaics.py -d [directory with mosaics]
    # For file: time python3 tile_mosaics.py -f [mosaic file] 

if __name__ == '__main__':
    main()

