
from ij import IJ
from ij.plugin.frame import RoiManager


try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
        from pathlib2 import Path
def run():
    roidir = '/Volumes/TRANSCEND/LauraLeopold/Wings-Rois/C/Roi/'
    maskdir = '/Volumes/TRANSCEND/LauraLeopold/Wings-Rois/C/Mask/'
    originaldir = '/Volumes/TRANSCEND/LauraLeopold/Wings-Rois/C/Original/'
    Raw_path = os.path.join(originaldir, '*tif')
    X = glob.glob(Raw_path)
    
    axes = 'YX'
    for fname in X:

      image = imread(fname)
      Name = os.path.basename(os.path.splitext(fname)[0])
      RoiName = roidir + Name
      print(fname, RoiName)
      
      rm = RoiManager.getInstance()
      if not rm:
        print "Please first add some ROIs to the ROI Manager"
        return
    impMask = IJ.createImage("Mask", "8-bit grayscale-mode", imp.getWidth(), imp.getHeight(), imp.getNChannels(), imp.getNSlices(), imp.getNFrames())
    IJ.setForegroundColor(255, 255, 255)
    rm.runCommand(impMask,"Deselect")
    rm.runCommand(impMask,"Fill")
    impMask.show()

run()