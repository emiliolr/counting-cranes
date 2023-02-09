# counting-more-cranes

This repository represents the continuation of the work of [Luz-Ricca et al. (2022)](https://doi.org/10.1002/rse2.301) and is in partnership with the William & Mary Institute for Integrative Conservation, the U.S. Fish & Wildlife Service, and the U.S. Geological Survey. This repository implements new methods relevant towards operationalization of automated counting of sandhill cranes using thermal aerial imagery and streamlines/updates the original codebase. 

## Initial Setup 

The Python version used for the original codebase is `Python 3.7`. Make sure to create a clean environment and install all required packages using `pip install -r requirements.txt`.

Pre-requisites for running the prediction pipeline (`full_pipeline.py`):
1. Model saves for Faster R-CNN or ASPDNet.
2. The mosaics to predict on (or single test images for local testing, e.g., from the final annotated dataset).