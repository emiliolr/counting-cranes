def run_pipeline():

    """
    A master function that assembles all pipeline elements.
    Inputs:
     -
    Outputs:
     - A master count for the input image set or mosaic (also saves run results)
    """

    #GENERAL NOTES:
    #  - push as many elements to functions as possible... boosts modularity
    #  - don't worry about making everything perfect/generalized from the get-go... it will take a lot of work to get this code into a final state as it's basically the final deliverable

    #CREATE MOSAIC: can circumvent by providing a pre-constructed mosaic
    #  - read in images
    #  - perform stitching
    #  - save final mosaic

    #TILE MOSAIC:
    #  - read in mosaic
    #  - tile according to protocol in "bird_dataset.py"
    #    - this includes pre-processing... no GT though!
    #  - save tiles (?)
    #    - seems like this will probably be necessary to avoid OOM errors

    #PREDICT ON TILES:
    #  - connect tiles to PyTorch dataloader
    #  - loop through the tiles and predict on each (keep running count)

    #CLEANUP:
    #  - save and return run results
    pass

if __name__ == '__main__':
    #ARGPARSER: use to collect run info from command line, then pass along to function
    #  - parent image root directory (optional)
    #  - mosaic FP (optional)
    #  - model save FP... as a .ckpt or .pth (required)
    #    - assume that this will be ASPDNet or Faster R-CNN
    #  - tiling method... w/ or w/o overlap (required)
    #  - file to write results to (required)
    #    - create if doesn't exist, add to otherwise ("MOSAIC_FP, runtime, final count")
    pass
