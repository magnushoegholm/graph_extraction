from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDF import TDF_LabelSequence, TDF_Label
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec
import uuid
import pprint
from graph_object import GraphObject
from true_center_tool import extract_true_centers
import numpy as np
import os

# Class to handle the uploaded file
class FileObject:
    def __init__(self, file=None, file_path=None):
        self.file = file
        self.file_path = file_path
        self.shape = None
        self.shape_tool = None
        self.doc = None
        self.load_file()
        self.bboxes = []
        self.centers = []
        self.true_centers = np.array([])

    def load_file(self):
        if self.file_path:
            print("There is a file path")
            file_path = self.file_path
        else:
            print("There is no file path")
            # Save the uploaded file to a temporary location
            file_path = f"/tmp/{self.file.filename}"
            self.file.save(file_path)
            print("Saved file to", file_path)
            self.file_path = file_path
            print("The filepath is: " + self.file_path)
        
        # Create a document
        self.doc = TDocStd_Document("pythonocc-doc-step-import")

        # Load the STEP file using STEPCAFControl_Reader
        step_reader = STEPCAFControl_Reader()
        status = step_reader.ReadFile(file_path)
        
        print("Loading file: ", file_path)

        # Check if the file has been loaded successfully
        if status == IFSelect_RetDone:
            step_reader.Transfer(self.doc)
            self.shape_tool = XCAFDoc_DocumentTool.ShapeTool(self.doc.Main())
            labels = TDF_LabelSequence()
            self.shape_tool.GetFreeShapes(labels)
            if labels.Length() > 0:
                self.shape = self.shape_tool.GetShape(labels.Value(1))
            print("File has been loaded successfully")
        else:
            raise Exception("Failed to load file")

    def get_num_components(self):
        if self.shape is None:
            return "No file loaded"
        
        explorer = TopExp_Explorer(self.shape, TopAbs_SOLID)
        num_components = 0
        while explorer.More():
            num_components += 1
            explorer.Next()
        return f"File has {num_components} components"
    
    def get_bounding_box(self, shape):
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox, True)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        # Calculate the center of the bounding box
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        z_center = (zmin + zmax) / 2
        
        # Calculate the dimensions of the bounding box
        x_dim = xmax - xmin
        y_dim = ymax - ymin
        z_dim = zmax - zmin
        
        # Return the bounding box coordinates and the center
        return (xmin, ymin, zmin, xmax, ymax, zmax), (x_center, y_center, z_center)

    def get_graph_data_for_user(self, socketio, username, include_relationships=True):
        """Get graph data with user-specific socket emissions"""
        print(f"Getting component data for user: {username}")
        graph = GraphObject()
        assembly_data = {}
        
        # Make sure temp directory exists
        if not os.path.exists("temp_stl_files"):
            os.makedirs("temp_stl_files")
        
        try:
            # Only extract true centers if we need relationships (for accurate edge computation)
            if include_relationships:
                print("Getting true centers for relationship analysis...")
                if socketio:
                    socketio.emit('status', {
                        'message': f'Extracting centers for relationship analysis..'
                    }, room=username)
                self.true_centers = extract_true_centers(self.file_path)
            else:
                print("Skipping true center extraction (components only)")
                if socketio:
                    socketio.emit('status', {
                        'message': 'Extracting components (fast mode)..'
                    }, room=username)
                self.true_centers = np.array([])  # Empty array for components-only mode
            
            def get_sub_shapes(lab, parent_uid=None):
                uid = str(uuid.uuid4())
                name = lab.GetLabelName()
                
                #print("Name: ", name)
                
                shape = self.shape_tool.GetShape(lab)
                
                props = GProp_GProps()
                brepgprop.VolumeProperties(shape, props)
                volume = props.Mass()
                brepgprop.SurfaceProperties(shape, props)
                surface_area = props.Mass()            
                shape_type = "unknown"
                
                if self.shape_tool.IsAssembly(lab):
                    shape_type = "Assembly"
                    l_c = TDF_LabelSequence()
                    self.shape_tool.GetComponents(lab, l_c)
                    for i in range(l_c.Length()):
                        label = l_c.Value(i + 1)
                        if self.shape_tool.IsReference(label):
                            label_reference = TDF_Label()
                            self.shape_tool.GetReferredShape(label, label_reference)
                            get_sub_shapes(label_reference, uid)

                elif self.shape_tool.IsSimpleShape(lab):
                    # Get the bounding box of the shape
                    bounding_box, center = self.get_bounding_box(shape)
                    self.bboxes.append(bounding_box)
                    self.centers.append(center)
                    
                    # Only apply true center transformation if we extracted true centers
                    if include_relationships and len(self.true_centers) > 0:
                        # Calculate the translation vector using true centers
                        true_center_index = len(self.centers) - 1
                        if true_center_index < len(self.true_centers):
                            translation_vector = np.array(self.true_centers[true_center_index]) - np.array(center)
                            
                            # Only apply the transformation if there is a difference in the centers
                            if not np.allclose(translation_vector, [0, 0, 0]):
                                # Create a transformation
                                trsf = gp_Trsf()
                                trsf.SetTranslation(gp_Vec(*translation_vector))
                                
                                # Apply the transformation
                                shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
                                
                                # Update the bounding box and center with the transformed shape
                                bounding_box, center = self.get_bounding_box(shape)
                                self.bboxes[-1] = bounding_box
                                self.centers[-1] = center
                    
                    # Add the shape to the graph
                    graph.add_component(f"{uid}", shape, bounding_box, name, position=None)
                    
                    shape_type = "Component"
                    if socketio:
                        socketio.emit('status', {
                            'message': f'Loaded component {name}'
                        }, room=username)
                    
                assembly_data[uid] = {
                    "name": name,
                    "type": shape_type,
                    "volume": float(volume),
                    "surface_area": float(surface_area),
                    "parent": parent_uid,
                    "position": None
                }
                
            def _get_shapes():
                labels = TDF_LabelSequence()
                self.shape_tool.GetFreeShapes(labels)
                for i in range(labels.Length()):
                    root_item = labels.Value(i + 1)
                    get_sub_shapes(root_item)
                    
            _get_shapes()
            
            # Only compute edges if relationships are requested
            if include_relationships:
                print("Computing edges")
                if socketio:
                    socketio.emit('status', {
                        'message': 'Computing edges..'
                    }, room=username)
                graph.compute_edges_for_user(socketio, username)
            else:
                print("Skipping edge computation (components only)")
                if socketio:
                    socketio.emit('status', {
                        'message': 'Skipping relationship analysis (components only)'
                    }, room=username)
            
            # graph.create_component_clusters()
            
            graph.update_component_positions()
            
            # Keep the exact same data structure as the original code
            nodes_dict = {k: v[1:] for k, v in graph.components.items()}
            
            # Only include edges if relationships were computed
            edges_dict = dict(graph.edges) if include_relationships else {}

            return {"Assembly data": assembly_data, "Components": nodes_dict, "Edges": edges_dict}
        except Exception as e:
            print(f"Error during graph data extraction: {e}")
            
            # In case of error, ensure cleanup still happens
            if hasattr(graph, 'cleanup_temp_files'):
                graph.cleanup_temp_files()
            raise

    def get_graph_data(self, socketio, include_relationships=True):
        """Original method - calls new user-specific method with None username"""
        return self.get_graph_data_for_user(socketio, None, include_relationships)

# Example usage
if __name__ == "__main__":
    file_path = r"colorful example model.step"
    file_object = FileObject(file_path=file_path)
    
    graph_data = file_object.get_graph_data(None)
    # pprint.pprint(graph_data)

    
    # Compare centers to true centers
    centers = np.array(file_object.centers)
    true_centers = file_object.true_centers
    
    print("Centers: ")
    print(centers)
    print("True centers: ")
    print(true_centers)
    
    # Save the graph data to a txt file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, "graph_data.txt")
    with open(output_file_path, 'w') as output_file:
        pprint.pprint(graph_data, stream=output_file)
    print(f"Graph data saved to {output_file_path}")


