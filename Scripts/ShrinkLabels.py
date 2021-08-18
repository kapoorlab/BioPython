#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tifffile import imread, imwrite
import os
import glob
from tqdm import tqdm
from tifffile import imread
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage.morphology import binary_erosion
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops


# In[2]:


def erode_labels(segmentation, erosion_iterations):
    # create empty list where the eroded masks can be saved to
    list_of_eroded_masks = list()
    regions = regionprops(segmentation)
    def erode_mask(segmentation_labels, label_id, erosion_iterations):
        
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = ndimage.binary_erosion(only_current_label_id, iterations = erosion_iterations)
        relabeled_eroded = np.where(eroded == 1, label_id, 0)
        return(relabeled_eroded)

    for i in range(len(regions)):
        label_id = regions[i].label
        list_of_eroded_masks.append(erode_mask(segmentation, label_id, erosion_iterations))

    # convert list of numpy arrays to stacked numpy array
    final_array = np.stack(list_of_eroded_masks)

    # max_IP to reduce the stack of arrays, each containing one labelled region, to a single 2D np array. 
    final_array_labelled = np.sum(final_array, axis = 0)
    return(final_array_labelled)


# In[ ]:


sourcedir = '/Users/aimachine/sample_segmented_data/Orignal_and_Ground_truth/RealMask/'
resultsdir = '/Users/aimachine/sample_segmented_data/Orignal_and_Ground_truth/BinaryMask/'
Path(resultsdir).mkdir(exist_ok = True)
Raw_path = os.path.join(sourcedir, '*tif')
X = glob.glob(Raw_path)
erosion_iterations = 2
for fname in X:

    Image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    ErodedImage =  erode_labels(Image, erosion_iterations)
    ErodedImage = ErodedImage > 0
    imwrite((resultsdir + Name + '.tif') , ErodedImage.astype('uint16'))


# In[ ]:




