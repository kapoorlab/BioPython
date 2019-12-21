#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:10:31 2019

@author: aimachine
"""

import collections
import numpy as np
import warnings
from tifffile import imsave
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from six.moves import reduce
from skimage.measure import label
from skimage import measure
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import remove_small_objects
from skimage.segmentation import  relabel_sequential
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage.filters import median_filter, gaussian_filter, maximum_filter
from scipy.ndimage.morphology import  binary_dilation
from skimage.filters import threshold_local, threshold_mean, threshold_otsu
from skimage.feature import canny
from skimage.filters import gaussian
from skimage import segmentation
import scipy.stats as st
from skimage.util import invert as invertimage
from scipy import ndimage as ndi
import napari


def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")



    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

def ConvolveGaussian(array, size, sigma):
    
    interval = (2*sigma+1.)/(size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    convolvedarray = ndi.convolve(array, kernel)
    
    return convolvedarray

def Annotate(Raw, SegImage):
    
    with napari.gui_qt():
        viewer = napari.view_image(Raw, name = 'ThreeDimage')
        viewer.add_image(SegImage)
        pts_layer = viewer.add_points(size = 5)
        pts_layer.mode = 'add'
    return pts_layer       
          
       
        
def showImageNapari(Raw,Image, rgb = False):
    with napari.gui_qt():
        
        viewer = napari.view_image(Image, rgb = rgb)
        viewer.add_image(Raw)
    

def normalizeZeroOne(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x
    
def BinaryDilation(Image, iterations = 1):

    DilatedImage = binary_dilation(Image, iterations = iterations) 
    
    return DilatedImage
    
def SelectSlice(Image, axis = 0, slicenumber = 0):

   assert len(Image.shape) >=3
   
   SmallImage = Image.take(indices = range(slicenumber - 1, slicenumber), axis = axis)
   
   return SmallImage
   
def MidSlices(Image, axis = 0, slices = 2):
    
    assert len(Image.shape) >=3
    
    SmallImage = Image.take(indices = range(Image.shape[axis]//2 - slices, Image.shape[axis]//2 + slices), axis = axis)
    
    MaxProject = np.amax(SmallImage, axis = axis)
        
    return MaxProject

def MaxProjection(Image, axis = 0):
    
    assert len(Image.shape) >= 3
    
    MaxProject = np.amax(Image, axis = axis)
        
    return MaxProject

def Embryo_plot(Time, Number):

    
    fig, ax = plt.subplots() 
    ax.plot(Time, Number, '-b', alpha=0.6,
        label='Embryos over time')
    x_min, x_max = ax.get_xlim()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number')
    plt.show()

def VarianceFilterTime(Image, kernel = (3,3)):
    
    
    VarImage = np.zeros([Image.shape[0], Image.shape[1], Image.shape[2]])
    MeanImage = np.zeros([Image.shape[0], Image.shape[1], Image.shape[2]])
    MeanSqImage = np.zeros([Image.shape[0], Image.shape[1], Image.shape[2]])
    for t in range(0,Image.shape[0]):
       MeanImage[t,:] = ndi.uniform_filter(Image[t,:], kernel)
       MeanSqImage[t,:] = ndi.uniform_filter(Image[t,:]**2, kernel)
       VarImage[t,:] = MeanSqImage[t,:] - MeanImage[t,:]**2
    
    return VarImage


    
    

def BackGroundCorrection3D(Image, sigma):
    
    
    Blur = np.zeros([Image.shape[0],Image.shape[1], Image.shape[2] ])
    Corrected = np.zeros([Image.shape[0],Image.shape[1], Image.shape[2] ])
    for j in range(0, Image.shape[0]): 
     Blur[j,:] = gaussian(Image[j,:].astype(float), sigma)
     Corrected[j,:] = Image[j,:] - Blur[j,:]
    
    
    return Corrected

def BackGroundCorrection2D(Image, sigma):
    
    
     Blur = gaussian(Image.astype(float), sigma)
     
     
     Corrected = Image - Blur
     
     return Corrected

def MedianFilter(Image,sigma):
    
    MedianFilter = median_filter(Image.astype(float), sigma)

    return MedianFilter

def count_white(Image):
    
    Area = 0
    for j in range(0, Image.shape[0]):
        for k in range(0, Image.shape[1]):
           if Image[j,k] > 0:
               Area = Area + 1

    return Area

def LocalThreshold3D(Image, boxsize, offset = 0, size = 10):
   
    Binary = np.zeros([Image.shape[0],Image.shape[1], Image.shape[2] ])
    Clean = np.zeros([Image.shape[0],Image.shape[1], Image.shape[2] ])
    for j in range(0, Image.shape[0]): 
        adaptive_thresh = threshold_local(Image[j,:], boxsize, offset=offset)
        Binary[j,:]= Image[j,:] > adaptive_thresh
        Clean [j,:]=  remove_small_objects(Binary[j,:], min_size=size, connectivity=4, in_place=False)
    return Clean

def LocalThreshold2D(Image, boxsize, offset = 0, size = 10):
    
    
    adaptive_thresh = threshold_local(Image, boxsize, offset=offset)
    Binary  = Image > adaptive_thresh
    Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Clean

def OtsuThreshold2D(Image, size = 10):
    
    
    adaptive_thresh = threshold_otsu(Image)
    Binary  = Image > adaptive_thresh
    Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Clean


def Canny3D(Image, sigma = 0.01):
    
    Canny = np.zeros([Image.shape[0],Image.shape[1], Image.shape[2] ])
    for j in range(0, Image.shape[0]): 
       Canny[j,:] = canny(Image[j,:], sigma)
    
    return Canny


def Canny(Image, sigma = 1):
    
    Canny = canny(Image, sigma)
    
    return Canny

def MakeLabels(image, metric='chessboard'):
    
  image = BinaryDilation(image)
  image = invertimage(image)
   
  labelimage = label(image)  

    
  labelclean = remove_big_objects(labelimage, max_size = 5000)  

  nonormimg, forward_map, inverse_map = relabel_sequential(labelclean) 
  #nonormimg = maximum_filter(nonormimg, 5)  
  return nonormimg  

def normalizeFloatZeroOne(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalizer(x, mi, ma, eps = eps, dtype = dtype)

def normalizer(x, mi , ma, eps = 1e-20, dtype = np.float32):


    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """


    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        x = normalizeZeroOne(x)
    return x
   
def NormalizeFloat(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[slices]
        # add dummy axes present in 'to'
        for i,a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x,-1)
                fr += a

    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def _raise(e):
    raise e
def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)


def invert(image):
    
    MaxValue = np.max(image)
    MinValue = np.min(image)
    image[:] = MaxValue - image[:] + MinValue
    
    return image

def ConnecProbability(img, minsize):
    
    binary = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    label = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(0, img.shape[0]):
        
      thresh = threshold_otsu(img[i,:])
    
      binary[i,:]  =  img[i,:] > thresh
      
      binary[i,:] = binary[i,:].astype('bool')
      
      binary[i,:] = binary_dilation(binary[i,:])
      
      label[i,:] = binary_fill_holes(binary[i,:] )
      label[i,:] = remove_small_objects(label[i,:].astype('bool'), min_size=minsize, connectivity=4, in_place=False)
     
      label[i,:] = normalizeBinaryMinMax(label[i,:], 0, 255) 
        
    return label

def normalizeBinaryMinMax(x, mi, ma,axis = None, clip = False, dtype = np.float32):
        """ Normalizing an image between min and max """
      
        x = mi + ((x ) ) * ma
        if clip:
               x = np.clip(x, 0 , 1)
        
        return x 
def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes

def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])

def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled
    
def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    axes: str
        Axes of ``img``
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """
    axes = axes_check_and_normalize(axes,img.ndim,disallowed='S')

    # convert to imagej-compatible data type
    t = img.dtype
    if   'float' in t.name: t_new = np.float32
    elif 'uint'  in t.name: t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif 'int'   in t.name: t_new = np.int16
    else:                   t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))

    # move axes to correct positions for imagej
        img = move_image_axes(img, axes, 'TZCYX', True)

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)
def WatershedImage(image,   kernel_sizeX, kernel_sizeY, kernel_sizeZ = None, min_distance = 10, sign = 'minus'):
    
    coordinates = peak_local_max(image, min_distance=min_distance)
    
    #print(coordinates, image[coordinates[:,0], coordinates[:,1]])
          
    coordinates_int = np.round(coordinates).astype(int)
    markers_raw = np.zeros_like(image)      
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(coordinates))
    #raw markers might be in a little watershed "well".
    markers = morphology.dilation(markers_raw, morphology.disk(10))

    if sign == 'minus':
     labels = segmentation.watershed(-image, markers)
    if sign == 'plus':
     labels = segmentation.watershed(image, markers)   
    nonormimg, forward_map, inverse_map = relabel_sequential(labels)    
    return nonormimg
    
def WatershedImageMarker(image, binary_fill, kernel_sizeX, kernel_sizeY, kernel_sizeZ = None, minsize = 10):
    
    if kernel_sizeZ is not None:
        kernel_size = kernel_sizeX, kernel_sizeY, kernel_sizeZ
    else:
        kernel_size = kernel_sizeX, kernel_sizeY
    
    local_maxi = peak_local_max((image), indices=False, footprint=np.ones((kernel_size)),labels=binary_fill)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-image, markers, mask=binary_fill)
    nonormimg = remove_small_objects(labels, min_size= minsize, connectivity=8, in_place=False)
    nonormimg, forward_map, inverse_map = relabel_sequential(nonormimg)    
    return nonormimg