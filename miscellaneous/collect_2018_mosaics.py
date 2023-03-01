import os
import glob
import argparse

def get_mosaic_fps(args):
    pattern = os.path.join(args.mosaic_directory, '**', '*.jp2') 
    all_files = glob.glob(pattern, recursive = True) # grab all mosaics in this directory + all sub-directories
    all_fils = [os.path.abspath(f) for f in all_files]

    return all_files

if __name__ == '__main__':
    # Collect command line arguments
    parser = argparse.ArgumentParser()
    AGLs = [500, 1000, 1500, 2000, 2500, 3000, 3500]

    parser.add_argument('mosaic_directory', help = 'the directory containing all mosaics to extract', type = str)
    parser.add_argument('--include_agls', nargs = '+', help = 'years to include', type = int, default = AGLs)

    args = parser.parse_args()
    
    #  check that all passed in AGLs are valid for 2018 data
    for agl in args.include_agls:
        assert agl in AGLs, f'{agl} feet is not a valid AGL'

    # Extract all mosaic filepaths from mosaic directory
    all_files = get_mosaic_fps(args)

    # Write these to filepaths to a text file
    with open('mosaic_filepaths.txt', 'w') as f:
        for fp in all_files:
            f.write(fp + '\n')