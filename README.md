# counting-cranes

This repository contains the code for my research with William & Mary's Institute for Integrative Conservation (IIC), the U.S. Fish & Wildlife Service (USFWS), and the U.S. Geological Survey (USGS). Applying deep learning approaches from the object counting literature, I aim to streamline the monitoring process for sandhill cranes. 

The final deliverable for this project will be a data processing pipeline able to efficiently count cranes in thermal aerial imagery, with comparable error to the aerial transect method currently used for USFWS surveys. The pipeline has two major components: prediction and overlap resolution. The pipeline is designed to be modular to allow for some flexibility in the selected methods. This means that new methods can be inserted as necessary, to reflect advances in object counting approaches or better methods for overlap resolution.

1. Prediction

The prediction component handles the identification, localization, and enumeration of cranes. For the prediction portion of the pipeline, I experiment with both object detection (Faster R-CNN) and object counting (ASPDNet) approaches. The detection approach is more naive, but also more in line with previous efforts within conservation research. I test ASPDNet, a density estimation method, alongside Faster R-CNN to compare the performance of the two approaches.

2. Overlap Resolution

The overlap resolution component combats over-counting in the final estimate that arises due to front-to-back overlap between contiguous images in a flight line. This secondary component introduces an additional challenge to the task, as counts cannot simply be aggregated across all images in isolation. I experiment with methods that leverage geospatial metadata collected by practitioners at USFWS and USGS, as well as methods that are not geospatially aware. 

Make sure to explore the `README.md` files in the subdirectories (`counting-cranes/density_estimation`, for instance) to get more information about the source code that I am adapting from other studies.
