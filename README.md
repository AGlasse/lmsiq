# lmsiq
LMS image quality analysis

### Summary
Code to ingest Zemax images representing the ideal, as-designed and Monte-Carlo perturbed instances of the anticipated aberrated system. Output includes en-slitted energy and line profiles along both the spatial and spectral directions, along with the wavelength dependence of key image quality performance parameters.

### User Guide
The code is intended to be run by execution of file 'lmsiq.py'.  Its primary behaviour is then controlled using the '*reanalyse*' boolean variable.  Set to **True**, line spread and en-slitted energy profiles will be calculated for all images in a dataset for both the spatial and spectral directions.  The profiles will be written into folder ./results/dataset/profiles/*' and a summary file encoded with the IPC factor (eg 20221026_summary_ipc_01_300 has a factor of 1.3 % applied).
  If *reanalyse* = **False**, existing profiles are read in to create wavelengths dependent plots of key image quality and spectral resolution parameters.  

### Tasks
- Inter pixel capacitance - The code includes the functionality to convolve a 3 x 3 detector pixel kernel to represent IPC, nominally using a 1.3 % IPC factor for adjacent pixels.

- Intra-pixel response - IN WORK - Include functionality to model the sub-pixel gain variation, by applying a sub-pixel sampling with transmission which varies with position across the pixel.  Currently the model images are sampled at a 0.25 detector pixel interval, so we will interpolate them onto a finer grid as necessary. 

- Stability modelling - IN WORK - Model the variation of Gauss fitted line centre stability with source position to investigate the impact of sub-pixel sampling.  
