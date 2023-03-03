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

1. Sign into the W&M lab computers (if needed [create an account](https://accounts.cs.wm.edu/newuser_template)).
2. [Install Conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers): download the Linux 64-bit installer for python 3.8.
2. `scp` the installer to the lab machine.
3. Run `chmod +x[installer.sh]` to turn the file into an executable.
4. Install minconda with default settings.
5. Run `source miniconda3/etc/profile.d/conda.sh`. This will have to be done every time a new lab session is opened. 
6. Install the [nightly build of PyTorch](https://pytorch.org/get-started/locally/) using conda.
6. Install required packages using `pip install -r lab_requirements.txt`.
