# counting-more-cranes

This repository represents the continuation of the work of [Luz-Ricca et al. (2022)](https://doi.org/10.1002/rse2.301) and is in partnership with the William & Mary Institute for Integrative Conservation, the U.S. Fish & Wildlife Service, and the U.S. Geological Survey. This repository implements new methods relevant towards operationalization of automated counting of sandhill cranes using thermal aerial imagery and streamlines/updates the original codebase. 

## Initial Setup 

The Python version used for the original codebase is `Python 3.7`. Make sure to create a clean environment and install all required packages using `pip install -r requirements.txt`. I recommend developing in a `conda` environment to avoid issue--see [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for guidance.

Pre-requisites for running the prediction pipeline (`full_pipeline.py`):
1. Model saves for Faster R-CNN or ASPDNet.
2. The single mosaic (currently can only handle one at a time) to predict on (or single test images for local testing, e.g., from the final annotated dataset).

To run `final_pipeline.py`, use the command `python3 final_pipeline.py MOSAIC_FP MODEL_NAME MODEL_FP RESULTS_FP`. Arguments are explained below:
- `MOSAIC_FP`: the filepath for the mosaic (or single image) to predict on.
- `MODEL_NAME`: either "ASPDNet" or "faster_rcnn".
- `MODEL_FP`: the filepath for the pre-trained model.
- `RESULTS_FP`: the filepath for saving prediction results on the inputted mosaic. If the file doesn't exit, it will be created.

## Lab machine setup instructions 

1. SSH into the W&M lab computers (if needed, [create an account](https://accounts.cs.wm.edu/newuser_template)).
   - Check the Python version. The codebase is expecting `Python 3.8.x`
2. Install the virtualenv package using: `pip install virtualenv`. This may already be downloaded.
3. Set up a new Python virtual environment with: `virtualenv counting-more-cranes-env`.
4. Activate the virtual environment with: `. ./counting-more-cranes-env/bin/activate` (you must be in the directory _above_ the venv directory for this command).
5. Install the [nightly build of PyTorch](https://pytorch.org/get-started/locally/) for linux, using `pip` with CUDA 11.8.
6. Install required packages with: `pip install -r lab_requirements.txt`.
