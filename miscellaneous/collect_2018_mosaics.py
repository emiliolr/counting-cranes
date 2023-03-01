import os
import glob
import argparse

def get_mosaic_fps(args):
    # Extracting all valid files
    pattern = os.path.join(args.mosaic_directory, '**', '*.jp2') 
    all_files = glob.glob(pattern, recursive = True) # grab all mosaics in this directory + all sub-directories
    all_files = [os.path.abspath(f) for f in all_files] # turn into absolute paths

    # Subsetting only the requested AGLs 
    final_files = []
    for agl in args.include_agls:
        str_to_check = f'{agl}agl'
        valid_files = [f for f in all_files if str_to_check in os.path.basename(f)] 
        final_files.extend(valid_files)

    final_files = list(set(final_files))

    return final_files

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
    all_files = sorted(get_mosaic_fps(args))
    print(f'Found {len(all_files)} mosaics at requested AGLs ({", ".join([str(agl) for agl in args.include_agls])})')

    # Write these to filepaths to a text file
    with open('mosaic_filepaths.txt', 'w') as f:
        for fp in all_files:
            f.write(fp + '\n')