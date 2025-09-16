from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.BRepBndLib import brepbndlib
import numpy as np
def get_center(shape):
    """ Compute the bounding box of a given shape and return the center and dimensions """
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox, True)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    # Calculate center and dimensions
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    dimensions = (xmax - xmin, ymax - ymin, zmax - zmin)
    
    return center, dimensions
            
          

def extract_true_centers(step_file):
    """ Extract bounding boxes and labels from a STEP file and print them """
    print("Loading file: ", step_file)
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)

    if status != IFSelect_RetDone:
        print("Error: Cannot read STEP file.")
        return

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # Create a document and get the shape tool
    doc = TDocStd_Document("pythonocc-doc-step-import")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

    # Iterate over all parts in the STEP file
    it = TopoDS_Iterator(shape)
    true_centers = []
    counter = 0
    while it.More():
        sub_shape = it.Value()
        center, dimensions = get_center(sub_shape)
        
        counter += 1
        true_centers.append(center)
        it.Next()

    true_centers_array = np.array(true_centers)
    return true_centers_array

if __name__ == "__main__":

    step_file = r"C:\Users\magnu\OneDrive\DTU\00 Special course 2\step files\ip67.step"
    bounding_boxes1 = extract_true_centers(step_file)
    print(bounding_boxes1)
