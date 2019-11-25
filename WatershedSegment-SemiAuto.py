try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
        from pathlib2 import Path

try:
        import tempfile
        tempfile.TemporaryDirectory
except (ImportError, AttributeError):
       from backports import tempfile

import glob
import os
  
from tifffile import imread
import napari
from utils import NormalizeFloat, normalizeZeroOne
import matplotlib.pyplot as plt
import sys
from utils import save_tiff_imagej_compatible,WatershedImage
from utils import MedianFilter, LocalThreshold3D, LocalThreshold2D,OtsuThreshold2D, Canny3D, Canny,BackGroundCorrection3D,BackGroundCorrection2D
from utils import MaxProjection, VarianceFilterTime, MidSlices, showImageNapari,Embryo_plot, SelectSlice, BinaryDilation

import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d

from scipy import ndimage as ndi
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from scipy.ndimage import gaussian_filter
from skimage.morphology import watershed
from skimage.feature import peak_local_max, canny
from skimage import segmentation
from skimage.segmentation import find_boundaries, relabel_sequential
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes, binary_dilation
from skimage.exposure import rescale_intensity
from skimage import data, io, morphology
sourcedir = '/Users/aimachine/Documents/OzEmbryo/Klein/'
resultsdir = '/Users/aimachine/Documents/OzEmbryo/Klein/Results/'
Path(resultsdir).mkdir(exist_ok = True)
Raw_path = os.path.join(sourcedir, '*tif')
X = glob.glob(Raw_path)
axes = 'TYX'
All4Dimages = []
All5Dimages = []
All3Dimages = []
for fname in X:

    image = imread(fname)
    if len(image.shape) == 3:
        All3Dimages.append(image)
    if len(image.shape) == 4:
        All4Dimages.append(image)
    if len(image.shape) > 4:
        All5Dimages.append(image)
    
    print(fname, image.shape, len(image.shape), len(All3Dimages), len(All4Dimages), len(All5Dimages))
    #If you only have 3D image else skip this block

idx = 0
ThreeDimage = All3Dimages[idx]
ThreeDimage = NormalizeFloat(ThreeDimage, 1, 99.8)
for i in range(0,(ThreeDimage.shape[0])):
        ThreeDimage[i,:,:] =  normalizeZeroOne(ThreeDimage[i,:,:])
        

with napari.gui_qt():
  viewer = napari.view_image(ThreeDimage, name = 'ThreeDimage')
  
  
  
  




        
  blur_radius = (15,15)

  Size_remove = 40
  TwoDCorrectedThreeDimage = np.zeros([ThreeDimage.shape[0],ThreeDimage.shape[1], ThreeDimage.shape[2]]) 
  TwoDMedianThreeDimage = np.zeros([ThreeDimage.shape[0],ThreeDimage.shape[1], ThreeDimage.shape[2]]) 
  TwoDBinaryThreeDimage = np.zeros([ThreeDimage.shape[0],ThreeDimage.shape[1], ThreeDimage.shape[2]]) 
  TwoDDilatedThreeDimage = np.zeros([ThreeDimage.shape[0],ThreeDimage.shape[1], ThreeDimage.shape[2]]) 
  CannyThreeDimage = np.zeros([ThreeDimage.shape[0],ThreeDimage.shape[1], ThreeDimage.shape[2]]) 
  median_filter_radius = (8,8)
  for j in range(0, ThreeDimage.shape[0]):
 
    TwoDCorrectedThreeDimage[j,:] = BackGroundCorrection2D(ThreeDimage[j,:], blur_radius) 
    TwoDMedianThreeDimage[j,:] = MedianFilter(TwoDCorrectedThreeDimage[j,:], median_filter_radius)
    
    TwoDBinaryThreeDimage[j,:] = OtsuThreshold2D(TwoDMedianThreeDimage[j,:], size = Size_remove)
    CannyThreeDimage[j,:] = Canny(TwoDBinaryThreeDimage[j,:], sigma = 3)
    TwoDDilatedThreeDimage[j,:] = BinaryDilation(TwoDBinaryThreeDimage[j,:], iterations = 15)
   
  pts_layer = viewer.add_points(size = 5)
  pts_layer.mode = 'add'
  # annotate the background and all the coins, in that order
  coordinates = pts_layer.data
  coordinates_int = np.round(coordinates).astype(int)
  markers_raw = np.zeros_like(ThreeDimage)
  markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(coordinates))
  # raw markers might be in a little watershed "well".
  markers = morphology.dilation(markers_raw, morphology.disk(5))
  segments = segmentation.watershed(-TwoDDilatedThreeDimage, markers=markers)
  labels_layer = viewer.add_labels(segments - 1)  # make background 0          