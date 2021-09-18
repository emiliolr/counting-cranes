from tifffile import tifffile as tif
import argparse

#Getting the mosaic filepath from command line input
parser = argparse.ArgumentParser()
parser.add_argument('mosaic_fp', help = 'file path for mosaic')
args = parser.parse_args()

#Reading in the file... its dtype is uint16
im = tif.imread(args.mosaic_fp)

#Converting to 8-bit
im8 = im.astype(np.uint8)

#Saving the new 8-bit TIF
tif.imsave(args.mosaic_fp + '_8bit', im8)
