import xml.etree.cElementTree as et
import os
import numpy as np
import pandas as pd
import csv
from skimage import measure
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider, QComboBox
from tqdm import tqdm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math
import matplotlib.pyplot as plt

Boxname = 'TrackBox'

def Velocity(Source, Target):
    
    
    ts,zs,ys,xs = Source
    
    tt,zt,yt,xt = Target
    
    Velocity = (float(zs) - float(zt)) * (float(zs) - float(zt)) + (float(ys) - float(yt)) * (float(ys) - float(yt)) + (float(xs) - float(xt)) * (float(xs) - float(xt))
    
    return math.sqrt(Velocity)/ max((float(tt)-float(ts)),1)
    

def ImportTrackmateXML(xml_path, Segimage, image = None):
    
        Name = os.path.basename(os.path.splitext(xml_path)[0])
        savedir = os.path.dirname(xml_path)
        root = et.fromstring(open(xml_path).read())
          
        filtered_track_ids = [int(track.get('TRACK_ID')) for track in root.find('Model').find('FilteredTracks').findall('TrackID')]
        
        #Extract the tracks from xml
        tracks = root.find('Model').find('AllTracks')
        #Extract the cell objects from xml
        Spots = root.find('Model').find('AllSpots') 
        
        #Make a dictionary of the unique cell objects with their properties        
        Uniqueobjects = {}
        Uniqueproperties = {}
        for frame in Spots.findall('SpotsInFrame'):
            
            for Spot in frame.findall('Spot'):
                cell_id = int(Spot.get("ID"))
                Uniqueobjects[cell_id] = [cell_id]
                Uniqueproperties[cell_id] = [cell_id]
                Uniqueobjects[cell_id].append([Spot.get('POSITION_T'),Spot.get('POSITION_Z'), Spot.get('POSITION_Y'), Spot.get('POSITION_X') ])
                Uniqueproperties[cell_id].append([Spot.get('INTENSITY')
                                                ,Spot.get('ESTIMATED_DIAMETER'), Spot.get('ESTIMATED_DIAMETER'), Spot.get('ESTIMATED_DIAMETER')]) 
                
                
        Tracks = []
        for track in tracks.findall('Track'):

            track_id = int(track.get("TRACK_ID"))
            Trackobjects = []
            Speedobjects = []
            SpotSourceTarget = []
            if track_id in filtered_track_ids:
                
                for edge in track.findall('Edge'):
                    
                   SpotSourceTarget.append([edge.get('Spot_SOURCE_ID'),edge.get('Spot_TARGET_ID'), edge.get('EDGE_TIME')])
                      
                
                #Sort the tracks by edge time  
                SpotSourceTarget = sorted(SpotSourceTarget, key = sortTracks , reverse = False)    
                for sourceID, targetID, EdgeTime in SpotSourceTarget:
                
                     Source = Uniqueobjects[int(sourceID)][1]
                     Target = Uniqueobjects[int(targetID)][1]
                
                     # TYXZ Location in track
                
                     Trackobjects.append(Source)
                     speed = Velocity(Source, Target)                 
                     Speedobjects.append(speed)
                # Create object trackID, T, Z, Y, X
                Tracks.append([track_id,Trackobjects, Speedobjects])
                
        #Sort tracks by their ID
        Tracks = sorted(Tracks, key = sortID, reverse = False)
               
        # Write all tracks to csv file as ID, T, Z, Y, X
        ID = []
        StartID = {}
        Tloc = []
        Zloc = []
        Yloc = []
        Xloc = []
        Speedloc = []
        SlocZ = []
        SlocY = []
        SlocX = []
        Vloc = []
        Iloc = []
        RegionID = {}
        VolumeID = {}
        locationID = {}        
        for trackid, Trackobjects, Speedobjects in Tracks:
            
             print('Computing Track TrackID:', trackid)  
             RegionID[trackid] = [trackid]
             VolumeID[trackid] = [trackid]
             locationID[trackid] = [trackid]
             StartID[trackid] = [trackid]
             Location = []
             Region = []
             Volume = []
             ID.append(trackid)
             Spot = Trackobjects[0]
             tstart = int(float(Spot[0]))
             
             Spot = Trackobjects[len(Trackobjects) - 1]
             tend = int(float(Spot[0]))
             StartID[trackid].append([tstart, tend])
             for i in tqdm(range(0,len(Trackobjects))):
                       Spot = Trackobjects[i]
                       VelocitySpot = Speedobjects[i]
                       t = int(float(Spot[0]))
                       z = int(float(Spot[1]))
                       y = int(float(Spot[2]))
                       x = int(float(Spot[3]))
                       speed = int(float(VelocitySpot))
                       Tloc.append(t)
                       Zloc.append(z)
                       Yloc.append(y)
                       Xloc.append(x)
                       Speedloc.append(speed)
                       if t < Segimage.shape[0]:
                               CurrentSegimage = Segimage[t,:]
                               CurrentLabel = CurrentSegimage[z,y,x]
                               if image is not None:
                                       Currentimage = image[t,:]
                                       properties = measure.regionprops(CurrentSegimage, Currentimage)
                               if image is None:
                                       properties = measure.regionprops(CurrentSegimage, CurrentSegimage)
                                       
                               
                               for prop in properties:
                                   
                                   if prop.label == CurrentLabel:
                                       
                                        sizeZ = abs(prop.bbox[0] - prop.bbox[3])
                                        sizeY = abs(prop.bbox[1] - prop.bbox[4])
                                        sizeX = abs(prop.bbox[2] - prop.bbox[5])
                                        Area = prop.area
                                        intensity = np.sum(prop.image)
                                        Vloc.append(Area)
                                        SlocZ.append(sizeZ)
                                        SlocY.append(sizeY)
                                        SlocX.append(sizeX)
                                        Iloc.append(intensity)
                                        Region.append([1,sizeZ, sizeY,sizeX])
                                        Volume.append([Area, intensity, speed])
                                        Location.append([t, z, y, x])
               
             locationID[trackid].append(Location)
             RegionID[trackid].append(Region)
             VolumeID[trackid].append(Volume)
                 
        df = pd.DataFrame(list(zip(ID,Tloc,Zloc,Yloc,Xloc, SlocZ, SlocY, SlocX, Vloc, Iloc, Speedloc)), index = None, 
                                              columns =['ID', 't', 'z', 'y', 'x', 'sizeZ', 'sizeY', 'sizeX', 'volume', 'intensity', 'speed'])

        df.to_csv(savedir + '/' + 'Extra' + Name +  '.csv')  
        df     
        
        # create the final data array: track_id, T, Z, Y, X
        
        df = pd.DataFrame(list(zip(ID,Tloc,Zloc,Yloc,Xloc)), index = None, 
                                              columns =['ID', 't', 'z', 'y', 'x'])

        df.to_csv(savedir + '/' + 'TrackMate' +  Name +  '.csv')  
        df

        return RegionID, VolumeID, locationID, Tracks, ID, StartID
    
class TrackViewer(object):
    
    
    def __init__(self, originalviewer, Raw, Seg, locationID, RegionID, VolumeID,  scale, ID, startID,canvas, ax):
        
        
        self.trackviewer = originalviewer
        self.Raw = Raw
        self.Seg = Seg
        self.locationID = locationID
        self.RegionID = RegionID
        self.VolumeID = VolumeID
        self.scale = scale
        self.ID = ID
        self.startID = startID
        self.layername = 'Trackpoints'
        self.layernamedot = 'Trackdot'
        #Side plots
        self.canvas = canvas
        self.ax = ax 

        self.UpdateTrack()
        self.plot() 
    def plot(self):
        
            for i in range(self.ax.shape[0]):
                 for j in range(self.ax.shape[1]):
                                   self.ax[i,j].cla()
            if self.ID!=Boxname:
                
                        self.ax[0,0].set_title("CellSize")
                        self.ax[0,0].set_xlabel("Time")
                        
                        self.ax[1,0].set_title("CellVolume")
                        self.ax[1,0].set_xlabel("Time")
                        
                        self.ax[0,1].set_title("CellVelocity")
                        self.ax[0,1].set_xlabel("Time")
                        
                        self.ax[1,1].set_title("CellIntensity")
                        self.ax[1,1].set_xlabel("Time")
                        #Execute the function    
                        
                        Location = self.locationID[int(float(self.ID))][1]
                        Volume =  self.VolumeID[int(float(self.ID))][1]
                        Region=  self.RegionID[int(float(self.ID))][1]
                        AllT = []
                        AllIntensity = []
                        AllArea = []
                        AllSpeed = []
                        AllSize = []
                        for i in range(0, len(Location)):
                            
                            t, z, y, x = Location[i]
                            area, intensity, speed = Volume[i]
                            sizeT, sizeZ, sizeY, sizeX = Region[i]
                            AllT.append(t)
                            AllArea.append(area)
                            AllIntensity.append(intensity)
                            AllSpeed.append(speed)
                            AllSize.append(sizeZ * (sizeY + sizeX))
                        self.ax[0,0].plot(AllT, AllSize)
                        self.ax[1,0].plot(AllT, AllArea)
                        self.ax[0,1].plot(AllT, AllSpeed)
                        self.ax[1,1].plot(AllT, AllIntensity)
                        
            self.canvas.draw()            
                        
    def UpdateTrack(self):
        
        
        
                if self.ID != Boxname:
                    
                        
                    
                        for layer in list(self.trackviewer.layers):
                           
                                 if self.layername == layer.name:
                                     self.trackviewer.layers.remove(layer)
                                 if self.layernamedot == layer.name:
                                     self.trackviewer.layers.remove(layer)
                                     
        
                
                        tstart, tend = self.startID[int(float(self.ID))][1]
                        self.trackviewer.dims.set_point(0, tstart)
                        Location = self.locationID[int(float(self.ID))][1]
                        Region = self.RegionID[int(float(self.ID))][1]
                        self.trackviewer.status = str(self.ID)
                        self.trackviewer.add_points(Location, scale = self.scale, face_color = 'transparent', symbol = 'ring',  edge_color = 'green', name=self.layername, size = Region)
                        self.trackviewer.add_points(Location, scale = self.scale, face_color = 'magenta', edge_color = 'green', name=self.layernamedot)
                        self.trackviewer.dims.ndisplay = 3
                         
                        

                    
            

                
def LiveTracks(Raw, Seg, scale, locationID, RegionID, VolumeID, ID, StartID):

    
    with napari.gui_qt():
            if Raw is not None:
                          
                          viewer = napari.view_image(Raw, scale = scale, name='Image')
                          Labels = viewer.add_labels(Seg, scale = scale, name = 'SegImage')
            else:
                          viewer = napari.view_image(Seg, scale = scale, name='SegImage')
            
            trackbox = QComboBox()
            trackbox.addItem(Boxname)
            
        
            for i in range(0, len(ID)):
                trackbox.addItem(str(ID[i]))
            
            figure = plt.figure(figsize = (4, 4))    
            multiplot_widget = FigureCanvas(figure)
            ax = multiplot_widget.figure.subplots(2,2)
            
            viewer.window.add_dock_widget(multiplot_widget, name = "TrackStats", area = 'right')
            multiplot_widget.figure.tight_layout()
            trackbox.currentIndexChanged.connect(lambda trackid = trackbox : TrackViewer(viewer, Raw, Seg, locationID, RegionID, VolumeID, scale, trackbox.currentText(), StartID,multiplot_widget, ax))
            viewer.window.add_dock_widget(trackbox, name = "TrackID", area = 'left')
            
def sortTracks(List):
    
    return int(float(List[2]))

def sortID(List):
    
    return int(float(List[0]))

    
def filter_Spots(Spots, name, value, isabove):
    if isabove:
        Spots = Spots[Spots[name] > value]
    else:
        Spots = Spots[Spots[name] < value]

    return Spots
