#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:10:31 2019

@author: Varun Kapoor
"""
import sys
import collections
import numpy as np
import warnings
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from scipy.fftpack import fftfreq
from PIL import Image
from bs4 import BeautifulSoup
import urllib.parse
from urllib.request import urlopen,Request
import random
from IPython.display import display
from tifffile import imsave
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from six.moves import reduce
from skimage.measure import label
from skimage import measure
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from skimage import morphology
import scipy.ndimage.filters as filters
from skimage.morphology import remove_small_objects
from skimage.segmentation import  relabel_sequential
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage.filters import median_filter, gaussian_filter, maximum_filter
from scipy.ndimage.morphology import  binary_dilation
from skimage.filters import threshold_local, threshold_mean, threshold_otsu
from skimage.feature import canny
from skimage import segmentation
import scipy.stats as st
from skimage.util import invert as invertimage
from scipy import ndimage as ndi
import napari
import os
import pandas as pd
import glob
import warnings
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit import Model
from numpy import exp, loadtxt, pi, sqrt
import math
from skimage.metrics import mean_squared_error
from scipy.signal import blackman
from scipy.fftpack import fft, ifft, fftshift
from scipy.fftpack import fftfreq
import numpy as np
from scipy.signal import find_peaks
from numpy import mean, sqrt, square
from matplotlib import cm
from skimage.filters import threshold_otsu, threshold_mean
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from tifffile import imread, imwrite
import cv2
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 1
warnings.filterwarnings("ignore") 
color = (255, 255, 0) 
thickness = 1
def Distance(locationA, locationB, ndim):
    distance = 0
    for i in range(ndim-1):
        
        distance = distance + (locationA[i] - locationB[i]) * (locationA[i] - locationB[i])
    return math.sqrt(distance)    
def sortXY(elem):
    
    return elem[0]
    
def ErrorMeassgeCats():
    Urls = ["https://www.reshot.com/free-stock-photos/cat/","https://www.reshot.com/free-stock-photos/kitten/"]
    url = random.choice(Urls)
    requester = {'User-Agent': 'Mozilla/5.0'}
    req=Request(url,headers=requester)
    u =urlopen(req)


    soup = BeautifulSoup(u.read())

    links = soup.find_all('a')

    images =[]
    for img in soup.findAll('img'):
      images.append(img.get('src'))
    img=random.choice(images)
    print(img)
    urllib.request.urlretrieve(img, "error.png")  
    try:
        img = Image.open("error.png")
        display(img)
        print("You have an error in the code, enjoy looking at this picture and contact your ground control to major Tom")
        #img.show()
    except:
        raise ValueError("No you are wrong")
    
    
def Tsurff(Raw, Seg, theta, TimeUnit):
    SegImage = imread(Seg)
    RawImage = imread(Raw)
    Locationtheta = []
    if len(SegImage.shape)==2:
        ndim = 2
    if len(SegImage.shape)==3:
        ndim = 3
    else:
        raise ValueError("Image dimension must be 2 or 3D")
        
    if ndim == 3:
        Clock = np.zeros_like(SegImage)
        TimeObject = {}
        TimeList = []
        for i in tqdm(range(0,SegImage.shape[0])):
            
            TimeList.append(i * TimeUnit)
            Locationtheta = {}
            TwoDImage = SegImage[i,:]
            SurfaceImage = find_boundaries(TwoDImage.astype('float32'))
            
           
            centroid, coords = findCentroid(SurfaceImage.astype('uint16'))
            coords = sorted(coords, key = sortXY, reverse = False)
            if i == 0:
                startcentroid, startcoords = findCentroid(SurfaceImage.astype('uint16'))
            toppoint = findTop(SurfaceImage.astype('uint16'), startcentroid, coords)
            bottompoint = findBottom(SurfaceImage.astype('uint16'), startcentroid, coords)
            
           
            
            startpoint = toppoint
            Locationtheta[0] = toppoint
            TimeAngleLocation = []
            Thetalist = AngleList(coords,startpoint, startcentroid)
            for angle in range(0, 360, theta):
                
                       pointlinedistance =  sys.float_info.min 
                       if angle == 0:
                            chosenlocation = startpoint
                       if angle > 0:
                          for (k,v) in Thetalist.items():
                                point = Thetalist[k]
                                if abs((float(k))-angle) < 1:
                                    pointdist = Distance(point, centroid, ndim)
                                    if pointdist > pointlinedistance:
                                        pointlinedistance = pointdist
                                        chosenlocation =  point
                                
                                    
                       Chosendistance = Distance(chosenlocation, startcentroid, ndim )
                       TimeAngleLocation.append([angle, Chosendistance])
                       Locationtheta[angle] = chosenlocation
                       cv2.circle(Clock[i,:], (int(chosenlocation[1]), int(chosenlocation[0])), 5,(255,0,0), thickness = -1 )
                       cv2.circle(Clock[i,:], (int(centroid[1]), int(centroid[0])), 5,(255,0,0), thickness = -1 )
                       cv2.putText(Clock[i,:], str(angle), (int(chosenlocation[1]), int(chosenlocation[0])), font,  
                               fontScale, color, thickness, cv2.LINE_AA)
            
            TimeObject[str(i)] = TimeAngleLocation
            ListMaps = []
            
            for givenangles in range(0, 360, theta):
                      AngleMap = {}
                      Dist = []
                      for (time, value) in TimeObject.items():
                           for angle, distance in value:
                            
                              if givenangles == angle:
                                   Dist.append(distance)
                      AngleMap[str(givenangles)] = Dist     
                                 
                      ListMaps.append(AngleMap)           
            
        
        return ListMaps, Clock, TimeList    
    
def DistAverage(Distlist, frames = 2):
    
    Moving_Average = []
    i = 0
    while i < len(Distlist) - frames + 1:

          this_window = Distlist[i:i + frames] 
          window_average = sum(this_window) / frames
          Moving_Average.append(window_average)
          i = i + 1
    for i in range(len(Distlist) - frames + 1,len(Distlist) ):
        Moving_Average.append(window_average)
            
    return Moving_Average 
                        
def AngleList(coords,startpoint, centroid):
    
    pointA = startpoint
    mintheta = sys.float_info.max
    returnpoint = None
    vector_1 = np.subtract(startpoint, centroid)

    Thetalist = {}
    for i in range(0, len(coords)):
            
            
            pointB = coords[i]
            vector_2 = np.subtract(pointB, centroid)

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            angle = (math.atan2(unit_vector_1[0],unit_vector_1[1]) - math.atan2(unit_vector_2[0],unit_vector_2[1]))* 180/math.pi
            
            if angle <0:
                angle+=360
            
            Thetalist[str(angle)] = pointB
            


    return Thetalist 

def LineAngled(centroid, radius, theta):
    
    y = centroid[0] + radius * math.sin(math.radians(theta))
    x = centroid[1] + radius * math.cos(math.radians(theta))

    return y, x

def distancepointline(slope, intercept, location):

    distance = abs(location[0] - slope * location[1] - intercept)/(math.sqrt(1+slope*slope))
    return distance


def findCentroid(image):
    
    Binary = image > 0
    props = regionprops(Binary.astype('uint16'))
    coords = props[0]['coords']
    centroid = props[0]['centroid']
    
    return centroid, coords

def findBottom(image, centroid, coords):
    
    Binary = image > 0
    maxY = sys.float_info.min
    for allcord in coords:
        #find the same x coordinate on the border as the center
        if(int(allcord[1]) == int(centroid[1])):
                if allcord[0] > maxY:
                   maxY = allcord[0]
                
    bottompoint = (maxY, centroid[1])            
    return bottompoint

def findTop(image, centroid, coords):
    
    Binary = image > 0
    minY = sys.float_info.max
    for allcord in coords:
        #find the same x coordinate on the border as the center
        if(int(allcord[1]) == int(centroid[1])):
                if allcord[0] < minY:
                   minY = allcord[0]
                
    toppoint = (minY, centroid[1])            
    return toppoint
def show_peak(onedimg, frequ, veto_frequ, threshold = 0.005):

    peaks, _ = find_peaks(onedimg, threshold = threshold)


    above_threshfrequ = []
    maxval = 0
    reqpeak =0
    for x in peaks:
      if(frequ[x] > veto_frequ):
        above_threshfrequ.append(x)
    for i in range(0,len(above_threshfrequ)):
      if onedimg[above_threshfrequ[i]] > maxval:
        maxval = onedimg[above_threshfrequ[i]]
        reqpeak = frequ[above_threshfrequ[i]] 
        
    frqY = reqpeak  
    
    return frqY

def CrossCorrelationStrip(imageA, imageB):
    
    PointsSample = imageA.shape[1] 
    stripA = imageA[:,0]
    stripB = imageB[:,0]
    stripCross = np.conjugate(stripA)* stripB
    Initial = 0
    x = []
    for i in range(stripA.shape[0]):
      x.append(i)
    for i in range(imageA.shape[1]):
        
        stripB = imageB[:,i]
        stripCross = np.conjugate(stripA)* stripB
        PointsSample += stripCross
        
    return PointsSample, x  


  
def RMSStrip(imageA, cal):
    rmstotal = np.empty(imageA.shape[0])
    PointsSample = imageA.shape[1]
    peri = range(0, int(np.round(imageA.shape[0] * cal)))
    for i in range(imageA.shape[0] - 1):
        stripA = imageA[i,:]
        RMS = sqrt(mean(square(stripA)))
        rmstotal[i] = RMS
        
    return [rmstotal, peri]    

def FFTStrip(imageA, Time_unit):
    ffttotal = np.empty(imageA.shape)
    PointsSample = imageA.shape[1]
    addedfft = 0 
    Blocks = []
    xf = fftfreq(PointsSample, Time_unit)         
    for i in range(imageA.shape[0]):
        stripA = imageA[i,:]
       
        fftstrip = fftshift(fft(stripA))
        ffttotal[i,:] = np.abs(fftstrip)
        addedfft += np.abs(fft(stripA))
        Blocks.append(np.abs(fft(stripA))[0:int(PointsSample//2)])
        
    return ffttotal, addedfft[0:int(PointsSample//2)], xf[0:int(PointsSample//2)], Blocks


def FFTSpaceStrip(imageA, Xcalibration):
    ffttotal = np.empty(imageA.shape)
    Blocks = []
    addedfft = 0 
    PointsSample = imageA.shape[0]
    
    xf = fftfreq(PointsSample, Xcalibration)         
    for i in range(imageA.shape[1]):
       
        stripA = imageA[:,i]
       
        fftstrip = fftshift(fft(stripA))
        ffttotal[:,i] = np.abs(fftstrip)
        addedfft += np.abs(fft(stripA))
        Blocks.append(np.abs(fft(stripA))[0:int(PointsSample//2)])
        
    return ffttotal, addedfft[0:int(PointsSample//2)], xf[0:int(PointsSample//2)], Blocks
    

    
    
    
def AnteriorPosterior(image, AnteriorStart, AnteriorEnd, PosteriorStart, PosteriorEnd, Xcalibration,savedir, threshold = 0.005):
    
    AnteriorVelocity = []
    PosteriorVelocity = []
    FrequAnteriorList = []
    FrequPosteriorList = []
    PeakAnterior = []
    PeakPosterior = []
    
    PointsSampleAnterior = image[AnteriorStart:AnteriorEnd,:].shape[0]
    xfAnterior = fftfreq(PointsSampleAnterior, Xcalibration)
    
    PointsSamplePosterior = image[PosteriorStart:PosteriorEnd,:].shape[0]
    xfPosterior = fftfreq(PointsSamplePosterior, Xcalibration)
     
    for i in range(0, image.shape[1]):
        
        
        AnteriorStrip = image[AnteriorStart:AnteriorEnd,i]
        FFTA =(np.abs((fft(AnteriorStrip))))
        FFTA = FFTA/np.amax(FFTA)
        peakA = show_peak(FFTA, xfAnterior, 0, threshold = threshold)
        if peakA > 0:
          PeakAnterior.append(peakA)            
        
        PosteriorStrip = image[PosteriorStart:PosteriorEnd,i]
        FFTP = (np.abs((fft(PosteriorStrip))))
        FFTP =FFTP/np.amax(FFTP)
        peakP = show_peak(FFTP, xfPosterior, 0, threshold = threshold)
        if peakP > 0:
           PeakPosterior.append(peakP)
        
        FrequAnteriorList.append(xfAnterior)
        FrequPosteriorList.append(xfPosterior)
        AnteriorVelocity.append(FFTA)
        PosteriorVelocity.append(FFTP)
    FrequAnteriorList = np.asarray(FrequAnteriorList)
    AnteriorVelocity = np.asarray(AnteriorVelocity)  
    
    MaxVelocityAnterior = np.amax(AnteriorVelocity, axis = 0)
    MaxFrequAnterior = np.amax(FrequAnteriorList, axis = 0) 
    
    FrequPosteriorList = np.asarray(FrequPosteriorList)
    PosteriorVelocity = np.asarray(PosteriorVelocity)  
    
    MaxVelocityPosterior = np.amax(PosteriorVelocity, axis = 0)
    MaxFrequPosterior = np.amax(FrequPosteriorList, axis = 0)  
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].plot( FrequAnteriorList, np.log(AnteriorVelocity), '-ro')
    ax[0].plot(MaxFrequAnterior, np.log(MaxVelocityAnterior))
    ax[0].set_xlabel('Momentum')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Anterior')
    ax[0].set_xlim([-0.1,1])
    
    ax[1].plot( FrequPosteriorList, np.log(PosteriorVelocity), '-ro')
    ax[1].plot(MaxFrequPosterior, np.log(MaxVelocityPosterior))
    ax[1].set_xlabel('Momentum')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Posterior')
    ax[1].set_xlim([-0.1,1])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    counts, bins = np.histogram(PeakAnterior)
    ax[0].hist(bins[:-1], bins, weights=counts)
    ax[0].set_xlabel('Momentum')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Anterior')        
    AnteriorMomentum = bins[np.argmax(counts)]
    
    counts, bins = np.histogram(PeakPosterior)
    ax[1].hist(bins[:-1], bins, weights=counts)
    ax[1].set_xlabel('Momentum')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Posterior') 
    PosteriorMomentum = bins[np.argmax(counts)]
    fig.savefig(savedir + 'AnteriorPosteriorMomentum' + '.png')
    return AnteriorMomentum, PosteriorMomentum  
        

def gaussian(x, amp, mu, std):
    return amp * exp(-(x-mu)**2 / std)



def BiModalgaussian(x, ampA, muA, stdA, ampB, muB, stdB):
    return ampA * exp(-(x-muA)**2 / stdA) + ampB * exp(-(x-muB)**2 / stdB)

def MSDAnalysis(CsvFile, savedir, nbins = 20, average = 10, time_unit = 10, displayfit = False ):
  
  Path(savedir).mkdir(exist_ok=True) 
  dataset = pd.read_csv(CsvFile)
 
  Name = os.path.basename(os.path.splitext(CsvFile)[0])
  print("Doing MSD analysis for file:")  
  print("Experiment analysis for :", Name)
  displacementX = dataset["X"][1:]
  displacementX = np.asarray(displacementX)
  
  deltaX = np.zeros_like(displacementX)

  for i in range(0, displacementX.shape[0] - average - 1):
      
      deltaX[i] = np.mean(displacementX[i + 1:i + average + 1]) - np.mean(displacementX[i:i+average])
  displacementY = dataset["Y"][1:]
  displacementY = np.asarray(displacementY)
  
  deltaY = np.zeros_like(displacementY)
  for i in range(0, displacementY.shape[0] - average - 1):
      
      deltaY[i] = np.mean(displacementY[i + 1:i+average + 1]) - np.mean(displacementY[i:i+ average])
  
  time = (dataset["Slice"][1:]) * time_unit  
  time = np.asarray(time)
    
  plt.plot(time, deltaX)   

  plt.xlabel("Time")
  plt.ylabel("DisplacementX")
  plt.savefig(savedir + Name + 'DisplacementX' + '.png')
  plt.show()   
  

  

  df = pd.DataFrame(list(zip(time.tolist(), deltaX.tolist())), columns =['time', 'displacementX'])
  df.to_csv(savedir + Name + 'DisplacementXT' +  '.csv', index = False) 
  df
  
  df = pd.DataFrame(list(zip(time.tolist(), deltaY.tolist())),columns =['time', 'displacementY'])
  df.to_csv(savedir + Name + 'DisplacementYT' +  '.csv', index = False) 
  
  # Histogram and Gaussian fit for deltaX
  meanX, stdX = norm.fit(deltaX) 
  countsX, binsX = np.histogram(deltaX, bins = nbins)
  binsX = binsX[:-1]  
  gmodel = Model(gaussian)
  Gauss = gmodel.fit(countsX, x=binsX, amp=np.max(countsX), mu=meanX, std=stdX)
  plt.hist(binsX, binsX, weights=countsX)
  if displayfit:
      plt.plot(binsX, Gauss.best_fit)
  plt.xlabel("DisplacementX")
  plt.ylabel("Counts")
  plt.savefig(savedir +Name+ 'HistDeltaX' + '.png')
  plt.show()
  
  df = pd.DataFrame(list(zip(binsX.tolist(), countsX.tolist(), Gauss.best_fit.tolist())),columns =['bins', 'counts', 'fit'])
  df.to_csv(savedir + Name + 'HistDeltaX' +  '.csv', index = False)  
  
  
  df = pd.DataFrame([[Gauss.fit_report()]],columns =['GaussFit parameters'])
  df.to_csv(savedir + Name + 'GaussFitsX' +  '.csv', index = False)   
  print('DisplacementX',Gauss.fit_report())
  
  plt.plot(time, deltaY)   

  plt.xlabel("Time")
  plt.ylabel("DisplacementY")
  plt.savefig(savedir + Name + 'DisplacementY' + '.png')
  plt.show()   
  
  
  # Histogram and Gaussian fit for deltaY
  meanY, stdY = norm.fit(deltaY) 
  countsY, binsY = np.histogram(deltaY, bins = nbins)
  binsY = binsY[:-1]  
  gmodel = Model(gaussian)
  Gauss = gmodel.fit(countsY, x=binsY, amp=np.max(countsY), mu=meanY, std=stdY)
  plt.hist(binsY, binsY, weights=countsY)
  if displayfit:
      plt.plot(binsY, Gauss.best_fit)
  plt.xlabel("DisplacementY")
  plt.ylabel("Counts")
  plt.savefig(savedir + Name +  'HistDeltaY' + '.png')
  plt.show()
  
  df = pd.DataFrame(list(zip(binsY.tolist(), countsY.tolist(), Gauss.best_fit.tolist())),columns =['bins', 'counts', 'fit'])
  df.to_csv(savedir + Name + 'HistDeltaY' +  '.csv', index = False)  
  
  
  df = pd.DataFrame([[Gauss.fit_report()]],columns =['GaussFit parameters'])
  df.to_csv(savedir + Name + 'GaussFitsY' +  '.csv', index = False)   
  print('DisplacementY',Gauss.fit_report())
  
  Totalmean = (meanX + meanY) / 2
  Totalstd = (stdX + stdY) / 2
  Totalcounts = countsX + countsY
  Totalbins = binsX + binsY
  gmodel = Model(gaussian)
  Gauss = gmodel.fit(Totalcounts, x=Totalbins, amp=np.max(Totalcounts), mu=Totalmean, std=Totalstd)
  plt.hist(Totalbins, Totalbins, weights=Totalcounts)
  if displayfit:
      plt.plot(Totalbins, Gauss.best_fit)
  plt.xlabel("DisplacementTotal")
  plt.ylabel("Counts")
  plt.savefig(savedir + Name +  'HistDeltaTotal' + '.png')
  plt.show()
  
  df = pd.DataFrame(list(zip(Totalbins.tolist(), Totalcounts.tolist(), Gauss.best_fit.tolist())),columns =['bins', 'counts', 'fit'])
  df.to_csv(savedir + Name + 'HistDeltaTotal' +  '.csv', index = False)  
  
  
  df = pd.DataFrame([[Gauss.fit_report()]],columns =['GaussFit parameters'])
  df.to_csv(savedir + Name + 'GaussFitsTotal' +  '.csv', index = False)   
  print('DisplacementTotal',Gauss.fit_report())
  
  
  
    

def AnteriorPosteriorTime(image, AnteriorStart, AnteriorEnd, PosteriorStart, PosteriorEnd, Tcalibration, savedir, threshold = 0.005):
    
    AnteriorVelocity = []
    PosteriorVelocity = []
    FrequAnteriorList = []
    FrequPosteriorList = []
    PeakAnterior = []
    PeakPosterior = []
    PointsSample = image.shape[1]
    xf = fftfreq(PointsSample, Tcalibration)
    
    
    for i in range(AnteriorStart, AnteriorEnd):
        
        
        Strip = image[i,:]
        FFTA =(np.abs((fft(Strip))))
        FFTA = FFTA/np.amax(FFTA)
        peakA = show_peak(FFTA, xf, 0, threshold = threshold)
        if peakA > 0:
          PeakAnterior.append(peakA)   
        
        FrequAnteriorList.append(xf)
        AnteriorVelocity.append(FFTA)
    FrequAnteriorList = np.asarray(FrequAnteriorList)
    AnteriorVelocity = np.asarray(AnteriorVelocity)  
    
    MaxVelocityAnterior = np.amax(AnteriorVelocity, axis = 0)
    MaxFrequAnterior = np.amax(FrequAnteriorList, axis = 0) 
    for i in range(PosteriorStart, PosteriorEnd):
        
        
        Strip = image[i,:]
        FFTP =(np.abs((fft(Strip))))
        FFTP = FFTP/np.amax(FFTP)
        peakP = show_peak(FFTP, xf, 0, threshold = threshold)
        if peakP > 0:
           PeakPosterior.append(peakP)
        
        FrequPosteriorList.append(xf)
        PosteriorVelocity.append(FFTP)
    FrequPosteriorList = np.asarray(FrequPosteriorList)
    PosteriorVelocity = np.asarray(PosteriorVelocity)  
    
    MaxVelocityPosterior = np.amax(PosteriorVelocity, axis = 0)
    MaxFrequPosterior = np.amax(FrequPosteriorList, axis = 0)         
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].plot( FrequAnteriorList, np.log(AnteriorVelocity), '-ro')
    ax[0].plot(MaxFrequAnterior, np.log(MaxVelocityAnterior))
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Anterior')
    ax[0].set_xlim([-0.001,0.01])
    
    ax[1].plot( FrequPosteriorList, np.log(PosteriorVelocity), '-ro')
    ax[1].plot(MaxFrequPosterior, np.log(MaxVelocityPosterior))
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Posterior')
    ax[1].set_xlim([-0.001,0.01])    
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    counts, bins = np.histogram(PeakAnterior)
    ax[0].hist(bins[:-1], bins, weights=counts)
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Anterior')        
    AnteriorFrequency = bins[np.argmax(counts)]
    
    
    counts, bins = np.histogram(PeakPosterior)
    ax[1].hist(bins[:-1], bins, weights=counts)
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Posterior')     
    PosteriorFrequency = bins[np.argmax(counts)]
    
    fig.savefig(savedir + 'AnteriorPosteriorFrequency' + '.png')
    return AnteriorFrequency, PosteriorFrequency  
            
   
def KymoMomentum(image,Xcalibration,savedir, threshold = 0.005):
    
    Velocity = []
    FrequList = []
    PeakValue = []
    PointsSample = image.shape[0]
    xf = fftfreq(PointsSample, Xcalibration)
    

    for i in range(0, image.shape[1]):
        
        
        Strip = image[:,i]
        FFT =(np.abs((fft(Strip))))
        FFT = FFT/np.amax(FFT)
        peak = show_peak(FFT, xf, 0, threshold = threshold)
        if peak > 0:
          PeakValue.append(peak)            
        
        
        FrequList.append(xf)
        Velocity.append(FFT)
        
    FrequList = np.asarray(FrequList)
    Velocity = np.asarray(Velocity)  
    
    MaxVelocity = np.amax(Velocity, axis = 0)
    MaxFrequ = np.amax(FrequList, axis = 0) 
    plt.plot( FrequList, np.log(Velocity), 'g')
    plt.plot(MaxFrequ, np.log(MaxVelocity))
    plt.xlabel('Momentum')
    plt.ylabel('Amplitude')
    plt.title('KymoKX')
    plt.xlim([-0.1,1])
    plt.show()
    counts, bins = np.histogram(PeakValue)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Momentum')
    plt.ylabel('Amplitude')
    plt.title('Peaks')        
    plt.show()
    Momentum = bins[np.argmax(counts)]
    plt.savefig(savedir + 'Momentum' + '.png')
    
    return Momentum  
        

def KymoTime(image, Tcalibration,savedir, threshold = 0.005):
    
    Velocity = []
    FrequList = []
    PeakValue = []
    PointsSample = image.shape[1]
    xf = fftfreq(PointsSample, Tcalibration)
    
    
    for i in range(0, image.shape[0]):
        
        
        Strip = image[i,:]
        FFT =(np.abs((fft(Strip))))
        FFT = FFT/np.amax(FFT)
        peak = show_peak(FFT, xf, 0, threshold = threshold)
        if peak > 0:
          PeakValue.append(peak)   
        
        FrequList.append(xf)
        Velocity.append(FFT)
   

    MaxVelocity = np.amax(Velocity, axis = 0)
    MaxFrequ = np.amax(FrequList, axis = 0)
    plt.plot( FrequList, np.log(Velocity), 'g')
    plt.plot(MaxFrequ, np.log(MaxVelocity))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('KymoWT')
    plt.xlim([-0.01,0.1])
    plt.show()
    counts, bins = np.histogram(PeakValue)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Peaks')
    plt.show()        
    Frequency = bins[np.argmax(counts)]
    plt.savefig(savedir + 'Frequency' + '.png')
    return Frequency      

def VelocityStrip(imageA, blocksize, Xcalibration):
    
    
   
     BlockVelocity = []
     diffimageA = np.zeros([imageA.shape[0], imageA.shape[1]])
     for i in range(0, imageA.shape[1] -  2 * blocksize):
       meanA = np.mean(imageA[:,i:i+blocksize], axis = 1)
       meanB = np.mean(imageA[:,i + blocksize:i + 2 * blocksize], axis = 1)
       diffimageA[:,i] = abs(meanB -  meanA) 
       
       

     return diffimageA
    
    

def DiffVelocityStrip(imageA, blocksize, Xcalibration):
    
    
   
     BlockVelocity = []
     for i in range(0, imageA.shape[0] - 2 * blocksize, blocksize):
       blockmean = np.mean(imageA[i:i+blocksize,:])
      
       BlockVelocity.append([i, blockmean])

     return BlockVelocity      
    
def PhaseDiffStrip(imageA):
    diff = np.empty(imageA.shape)
    value = np.empty(imageA.shape)
    for i in range(imageA.shape[0] - 1):
       
        diff[i, :] = imageA[i,:] - imageA[i + 1, :]
    return diff
    
def PhaseStrip(imageA):
    ffttotal = np.empty(imageA.shape)
    PointsSample = imageA.shape[1] 
    for i in range(imageA.shape[0]):
        stripA = imageA[i,:]
        
        fftstrip = (fft(stripA))
        ffttotal[i,:] = np.angle(fftstrip)
    return ffttotal


def sumProjection(image):
    sumPro = 0
    time = range(0, image.shape[1])
    time = np.asarray(time)
    for i in range(image.shape[0]):
        strip = image[i,:]
        sumPro += np.abs(strip) 
 
   
    return [sumPro, time]

def maxProjection(image):
    time = range(0, image.shape[1])
    time = np.asarray(time)
         

    return [np.amax(image, axis = 0), time]
    
#FFT along a strip
def doFilterFFT(image,Time_unit, filter):
   addedfft = 0 
   PointsSample = image.shape[1] 
   for i in range(image.shape[0]):
      if filter == True:   
       w = blackman(PointsSample)
      if filter == False:
       w = 1
      strip = image[i,:]
       
      fftresult = fft(w * strip)
      addedfft += np.abs(fftresult)  
   #addedfft/=image.shape[0]
   
   
   xf = fftfreq(PointsSample, Time_unit)
   
   
   return addedfft[1:int(PointsSample//2)], xf[1:int(PointsSample//2)]



def do2DFFT(image, Space_unit, Time_unit, filter):
    fftresult = fft(image)
    PointsT = image.shape[1]
    PointsY = image.shape[0]
    Tomega = fftfreq(PointsT, Time_unit)
    Spaceomega = fftfreq(PointsY, Space_unit)
    
    return fftresult

def do2DInverseFFT(image, Space_unit, Time_unit, filter):
    fftresult = ifft(image)
    PointsT = image.shape[1]
    PointsY = image.shape[0]
    Tomega = fftfreq(PointsT, Time_unit)
    Spaceomega = fftfreq(PointsY, Space_unit)
    
    return fftresult
def CrossCorrelation(imageA, imageB):
    crosscorrelation = imageA
    fftresultA = fftshift(fft(imageA))
    fftresultB = fftshift(fft(imageB))
    multifft = fftresultA * np.conj(fftresultB)
    crosscorrelation = fftshift(ifft(multifft))
    return np.abs(crosscorrelation) 


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

def SumProjection(Image, axis = 0):
    
    assert len(Image.shape) >= 3
    
    MaxProject = np.sum(Image, axis = axis)
        
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

def VarianceFilter(Image, kernel = (3,3)):
    
    
 
    MeanImage = ndi.uniform_filter(Image, kernel)
    MeanSqImage = ndi.uniform_filter(Image**2, kernel)
    VarImage = MeanSqImage - MeanImage**2
    
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

def WatershedLabels(image):
   image = BinaryDilation(image)
   image = invertimage(image)
   labelimage = label(image)
   labelimage =  filters.maximum_filter(labelimage, 4) 

   nonormimg, forward_map, inverse_map = relabel_sequential(labelimage) 


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

def CCLabels(image):
   image = BinaryDilation(image)
   labelimage = label(image)
   labelimage = ndi.maximum_filter(labelimage, size=4)
   
   nonormimg, forward_map, inverse_map = relabel_sequential(labelimage)  
    
def MakeLabels(image):
    
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
 
def save_8bit_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
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
    t = np.uint8
    t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))

    # move axes to correct positions for imagej
        img = move_image_axes(img, axes, 'TZCYX', True)

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)    
    




def watershed_binary(image, size, gaussradius, kernel, peakpercent):
 
 
 distance = ndi.distance_transform_edt(image)

 gauss = gaussian_filter(distance, gaussradius)

 local_maxi = peak_local_max(gauss, indices=False, footprint=np.ones((kernel, kernel)),
                            labels=image)
 markers = ndi.label(peakpercent * local_maxi)[0]
 labels = watershed(-distance, markers, mask=image)


 nonormimg = remove_small_objects(labels, min_size=size, connectivity=4, in_place=False)
 nonormimg, forward_map, inverse_map = relabel_sequential(nonormimg)    
 labels = nonormimg

 return labels   
def watershed_image_hough(image,  Xcalibration,Time_unit):
 distance = ndi.distance_transform_edt(image)

 
 local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((1, 1)),
                            labels=image)
 markers = ndi.label(local_maxi)[0]
 labels = watershed(-distance, markers, mask=image)

 nonormimg = remove_small_objects(labels, min_size=5, connectivity=4, in_place=False)
 nonormimg, forward_map, inverse_map = relabel_sequential(nonormimg)    
 labels = nonormimg


 Velocity = []
 Images = []
 Besty0 = []
 Besty1 = []
 # loop over the unique labels returned by the Watershed
 # algorithm
 for label in tqdm(np.unique(labels)):
      
      if label== 0:
            continue
     
      mask = np.zeros(image.shape, dtype="uint8")
      mask[labels == label] = 1
     
          
      h, theta, d = hough_line(mask)  
      img, besty0, besty1, velocity = show_hough_linetransform(mask, h, theta, d, Xcalibration, 
                               Time_unit)

      if np.abs(velocity) > 1.0E-5:  
       Velocity.append(velocity)
       Images.append(img)
       Besty0.append(besty0)
       Besty1.append(besty1)
 return Velocity    

def doubleplot(imageA, imageB, titleA, titleB):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.Spectral)
    ax[0].set_title(titleA)
    
    ax[1].imshow(imageB, cmap=cm.Spectral)
    ax[1].set_title(titleB)
    
    plt.tight_layout()
    plt.show() 
    
def tripleplot(imageA, imageB, imageC, titleA, titleB, titleC):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.Spectral)
    ax[0].set_title(titleA)
    
    ax[1].imshow(imageB, cmap=cm.Spectral)
    ax[1].set_title(titleB)
    
    ax[2].imshow(imageC, cmap=cm.Spectral)
    ax[2].set_title(titleC)
    
    plt.tight_layout()
    plt.show()
def show_hough_linetransform(img, accumulator, thetas, rhos, Xcalibration, Tcalibration,  save_path=None, File = None):
    import matplotlib.pyplot as plt


    
    bestpeak = 0
    bestslope = 0
    besty0 = 0
    besty1 = 0
    Est_vel = []
    for _, angle, dist in zip(*hough_line_peaks(accumulator, thetas, rhos)):
     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
     y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    
     pixelslope =   -( np.cos(angle) / np.sin(angle) )
     pixelintercept =  dist / np.sin(angle)  
     slope =  -( np.cos(angle) / np.sin(angle) )* (Xcalibration / Tcalibration)
     
    #Draw high slopes
     peak = 0;
     for index, pixel in np.ndenumerate(img):
            x, y = index
            vals = img[x,y]
            if  vals > 0:
                peak+=vals
                if peak >= bestpeak:
                    bestpeak = peak
                    bestslope = slope
                    besty0 = y0
                    besty1 = y1
   
    

    if save_path is not None and File is not None:
       plt.savefig(save_path + 'HoughPlot' + File + '.png')
    if save_path is not None and File is None:
        plt.savefig(save_path + 'HoughPlot' + '.png')
  
    
   

    return (img,besty0, besty1, bestslope)       
    


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