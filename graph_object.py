import trimesh
from scipy.spatial import cKDTree
import os
import numpy as np
from OCC.Extend.DataExchange import write_stl_file
import matplotlib.pyplot as plt
import networkx as nx
from string_clustering import StringClustering
import time
import atexit
import glob

# Class to handle the graph object
class GraphObject:
    def __init__(self):
        self.edges = {}
        self.components = {}
        # Ensure temp directory exists
        self.temp_dir = "temp_stl_files"
        self._ensure_temp_directory()
        
        # Register cleanup function to run when program exits
        atexit.register(self.cleanup_temp_files)
    
    def _ensure_temp_directory(self):
        """Ensure the temporary STL directory exists"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def cleanup_temp_files(self, age_minutes=None):
        """Clean up temporary STL files
        
        Args:
            age_minutes: If provided, only delete files older than this many minutes
        """
        try:
            if not os.path.exists(self.temp_dir):
                return
                
            current_time = time.time()
            for file_path in glob.glob(f"{self.temp_dir}/*.stl"):
                try:
                    # If age_minutes is specified, check file age
                    if age_minutes is not None:
                        file_age_minutes = (current_time - os.path.getmtime(file_path)) / 60
                        if file_age_minutes < age_minutes:
                            continue
                    
                    os.remove(file_path)
                    print(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete file {file_path}: {e}")
        except Exception as e:
            print(f"Error during temp file cleanup: {e}")
    
    def add_component(self, uid, shape, bounding_box, name, cluster=0, position=None): 
        self.components[uid] = [shape, [float(x) for x in bounding_box], name, position, cluster]
    
    # Function to create the component clusters based on the component names
    def create_component_clusters(self, threshold=0.6):
        try:
            clustering = StringClustering(threshold)
            
            component_names = [component[2] for component in self.components.values()]
            
            # Handle edge cases where clustering might fail
            if len(component_names) == 0:
                print("No components to cluster")
                return
            elif len(component_names) == 1:
                print("Only one component, assigning to cluster 0")
                for uid, component in self.components.items():
                    component[4] = 0
                return
            elif len(set(component_names)) == 1:
                print("All components have identical names, assigning to cluster 0")
                for uid, component in self.components.items():
                    component[4] = 0
                return
            
            clusters = clustering.cluster_strings(component_names)
            
            for cluster_id, component_names in clusters.items():
                for component_name in component_names:
                    for uid, component in self.components.items():
                        if component[2] == component_name:
                            component[4] = cluster_id
                            
        except Exception as e:
            print(f"Error during clustering: {e}")
            print("Falling back to assigning all components to cluster 0")
            # Fallback: assign all components to cluster 0
            for uid, component in self.components.items():
                component[4] = 0

    # Function to export the STL file    
    @staticmethod
    def export_stl(shape, filename):
        try:
            # Make sure that the output directory exists
            output_dir = os.path.dirname(filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            write_stl_file(shape, filename)
            
        except Exception as e:
            print(f"Failed to export STL file: {e}")
            raise
    
    # Function to filter out the bounding boxes and return the pairs of overlapping bounding boxes
    @staticmethod
    def bounding_box_filter(components, bb_threshold=0):
        
        # Function to check if two bounding boxes overlap
        def boxes_overlap(box1, box2, threshold):
            xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = box1
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = box2

            return (xmin1 - threshold <= xmax2 and xmax1 + threshold >= xmin2 and
                    ymin1 - threshold <= ymax2 and ymax1 + threshold >= ymin2 and
                    zmin1 - threshold <= zmax2 and zmax1 + threshold >= zmin2)

        # Make array to store pairs of overlapping bounding boxes
        bounding_box_pairs = []
        # Get all unique IDs
        uids = list(components.keys())
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                box1 = components[uids[i]][1]
                box2 = components[uids[j]][1]
                if boxes_overlap(box1, box2, bb_threshold):
                    bounding_box_pairs.append((uids[i], uids[j]))
        return bounding_box_pairs

    @staticmethod
    def create_mesh_from_shape(shape, uid, temp_dir="temp_stl_files"):
        stl_path = f"{temp_dir}/{uid}.stl"
        GraphObject.export_stl(shape, stl_path)
        mesh = trimesh.load(stl_path)
        return mesh
    
    # Function to visualize the signed distance points in a 3D plot
    # Used during the process of designing the constraint checking algorithm
    @staticmethod
    def plot_signed_distance_points(pts1, pts2, inside_pts1, inside_pts2, on_face_pts1, on_face_pts2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Points on the surface (green)
        on_face_pts = np.vstack((on_face_pts1, on_face_pts2))
        if len(on_face_pts) > 0:
            ax.scatter(on_face_pts[:, 0], on_face_pts[:, 1], on_face_pts[:, 2], c='g', label='On Face')

        # Inside points (black)
        if len(inside_pts1) > 0:
            ax.scatter(inside_pts1[:, 0], inside_pts1[:, 1], inside_pts1[:, 2], c='k', marker='x', label='Inside Mesh2')
        if len(inside_pts2) > 0:
            ax.scatter(inside_pts2[:, 0], inside_pts2[:, 1], inside_pts2[:, 2], c='k', marker='x', label='Inside Mesh1')

        # Remaining points (filter out on-face and inside points)
        mask1 = ~np.any(np.all(pts1[:, None] == np.vstack((on_face_pts1, inside_pts1)), axis=-1), axis=1)
        mask2 = ~np.any(np.all(pts2[:, None] == np.vstack((on_face_pts2, inside_pts2)), axis=-1), axis=1)


        remaining_pts1 = pts1[mask1]
        remaining_pts2 = pts2[mask2]

        ax.scatter(remaining_pts1[:, 0], remaining_pts1[:, 1], remaining_pts1[:, 2], c='r', label='Mesh1 Points')
        ax.scatter(remaining_pts2[:, 0], remaining_pts2[:, 1], remaining_pts2[:, 2], c='b', label='Mesh2 Points')

        # Adjust plot scale
        mid_x = (pts1[:, 0].max() + pts1[:, 0].min()) * 0.5
        mid_y = (pts1[:, 1].max() + pts1[:, 1].min()) * 0.5
        mid_z = (pts1[:, 2].max() + pts1[:, 2].min()) * 0.5

        max_range = np.array([pts1[:, 0].max() - pts1[:, 0].min(), 
                            pts1[:, 1].max() - pts1[:, 1].min(), 
                            pts1[:, 2].max() - pts1[:, 2].min()]).max() / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.show()
    
    # If a face intersection is found this function is used to check the directions of the intersection
    @staticmethod
    def check_directions(mesh1, mesh2, pts1, pts2, inside_threshold, outside_threshold):
        
        directions = [0, 0, 0, 0, 0, 0]
        
        for i in range(3):
            for j in range(2):
                
                # Create the translation vector
                translation = np.zeros(3)
                translation[i] = (j * 2 - 1) * outside_threshold # (j * 2 - 1) toggles between -1 and 1
                mesh1_copy = mesh1.copy()
                mesh1_copy.apply_translation(translation)
                
                # Check if translation lead to volumetric overlap
                sdf_values1 = -trimesh.proximity.signed_distance(mesh1_copy, pts2)
                sdf_values2 = -trimesh.proximity.signed_distance(mesh2, pts1)
                
                # Inside points (negative signed distance)
                inside_pts1 = pts1[sdf_values2 < -inside_threshold]
                inside_pts2 = pts2[sdf_values1 < -inside_threshold]
                
                if len(inside_pts1) > 0 or len(inside_pts2) > 0:
                    directions[i * 2 + j] = 1
        
        print("Directions are: ", directions)
        return directions
    
    # Function to check the signed distance between two meshes
    # Not to be confused with signed distance fields
    @staticmethod
    def check_signed_distance(mesh1, mesh2, pts1, pts2, inside_threshold, outside_threshold, plotting=False):
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # Use the PythonOCC function to compute the signed distance between mesh and points
        sdf_values1 = -trimesh.proximity.signed_distance(mesh1, pts2)
        sdf_values2 = -trimesh.proximity.signed_distance(mesh2, pts1)

        # Inside points (negative signed distance)
        inside_pts2 = pts2[sdf_values1 < -inside_threshold]
        inside_pts1 = pts1[sdf_values2 < -inside_threshold]

        # Points on the surface
        on_face_pts1 = pts1[(sdf_values2 > -inside_threshold) & (sdf_values2 < outside_threshold)]
        on_face_pts2 = pts2[(sdf_values1 > -inside_threshold) & (sdf_values1 < outside_threshold)]

        # If plotting is set, plot the points
        # This can not be set when the script is run in the background and the flask server is not running
        if plotting:
            GraphObject.plot_signed_distance_points(pts1, pts2, inside_pts1, inside_pts2, on_face_pts1, on_face_pts2)
        
        
        distance = min(sdf_values1.min(), sdf_values2.min())
        
        print("Distance is: ", distance)
        
        # If there are any inside points, the meshes intersect volumetrically and all directions are checked
        if len(inside_pts1) > 0 or len(inside_pts2) > 0:
            return -1, distance, [1,1,1,1,1,1]
        # If there are any points on the surface, the meshes intersect on the surface and the directions are checked
        elif len(on_face_pts1) > 0 or len(on_face_pts2) > 0:
            print("Checking directions..")
            return 0, distance, GraphObject.check_directions(mesh1, mesh2, pts1, pts2, inside_threshold, outside_threshold)
        # If there are no inside points or points on the surface, the meshes are not intersecting
        else:
            return 1, distance, None

    # Load the graph into networkX to be able to generate the spatial layout of the nodes so they look good in the visualization
    def update_component_positions(self):        
        
        G = nx.Graph()
        for uid in self.components.keys():
            G.add_node(uid)
        
        for uid, edges in self.edges.items():
            for edge in edges:
                G.add_edge(uid, edge["uid"])
        
        # Handle single component case
        if len(self.components) == 1:
            uid = list(self.components.keys())[0]
            self.components[uid][3] = [0.5, 0.5]  # Center position
            return
        
        # To add some relation between the physical model and the graph node position, we use the y-coordinate of the bounding boxes
        min_y = min(self.components[uid][1][1] for uid in self.components.keys())  # Get min bounding box Y
        max_y = max(self.components[uid][1][1] for uid in self.components.keys())  # Get max bounding box Y

        # Normalize initial positions based on bounding box Y-coordinates
        init_pos = {}
        for uid in self.components.keys():
            bbox = self.components[uid][1]
            y_normalized = (bbox[1] - min_y) / (max_y - min_y) if max_y > min_y else 0.5  # Avoid division by zero
            init_pos[uid] = np.array([np.random.rand(), y_normalized])  # Random X, controlled Y
        
        # Use spring_layout with initial positions
        pos = nx.spring_layout(G, k=0.7, pos=init_pos, seed=42)

        # Get all position values for scaling
        pos_values = list(pos.values())
        pos_min = np.min(pos_values, axis=0)
        pos_max = np.max(pos_values, axis=0)
        
        # Calculate range for each axis
        pos_range = pos_max - pos_min
        
        # Normalize final positions and flip the coordinate system (the front end uses a different coordinate system)
        for uid, position in pos.items():
            # Handle cases where all positions are the same (avoid division by zero)
            if np.any(pos_range == 0):
                # If positions are identical, use default positioning with small random offset
                scaled_position = np.array([0.5, 0.5]) + np.random.uniform(-0.1, 0.1, 2)
            else:
                # Normal scaling
                scaled_position = 0.2 + 0.6 * (position - pos_min) / pos_range
            
            flipped_position = [1 - scaled_position[0], 1 - scaled_position[1]]  # Flip the coordinates
            flipped_position[0] += np.random.uniform(-0.04, 0.04)  # Add small noise to x-coordinate
            flipped_position[1] += np.random.uniform(-0.01, 0.01)  # Add small noise to y-coordinate
            
            # Ensure positions stay within bounds
            flipped_position[0] = max(0.1, min(0.9, flipped_position[0]))
            flipped_position[1] = max(0.1, min(0.9, flipped_position[1]))
            
            self.components[uid][3] = [float(x) for x in flipped_position]  # Convert to list of floats

        labels = {uid: self.components[uid][2] for uid in self.components.keys()}
        
        # nx.draw(G, pos, labels=labels, with_labels=True)
        # plt.show()
    
    # Temporary function to flip the directions before I have implemented the correct logic
    def flip_directions(self, directions):
        if directions is not None:
            directions[0], directions[1] = directions[1], directions[0]
            directions[2], directions[3] = directions[3], directions[2]
            directions[4], directions[5] = directions[5], directions[4]
        return directions
        
        
    # Function to compute the edges between the components
    def compute_edges_for_user(self, socketio=None, username=None):
        """Compute edges with user-specific socket emissions"""
        # Clean up old temporary files (files older than 10 minutes)
        if socketio:
            socketio.emit('status', {
                'message': "Cleaning up old temporary files..."
            }, room=username)
        
        # Get the total number of components
        total_components = len(self.components)
        
        if total_components < 2:
            print("Not enough components to compute edges")
            if socketio:
                socketio.emit('status', {'message': "Not enough components to compute edges"})
            return

        # Filter out pairs of components that have overlapping bounding boxes
        bounding_box_pairs = self.bounding_box_filter(self.components)
        
        # Print the number of bounding box candidates out of all possible pairs
        print(f"Bounding box candidates: {len(bounding_box_pairs)} / {total_components * (total_components - 1) / 2}")

        # Create meshes for all components
        meshes = {}
        created_files = []
        
        try:
            if socketio:
                socketio.emit('status', {
                    'message': "Creating meshes.."
                }, room=username)
            
            print("Creating meshes..")
            
            for index, (uid, (shape, bbox, name, position, cluster)) in enumerate(self.components.items()):
                try:
                    meshes[uid] = self.create_mesh_from_shape(shape, uid, self.temp_dir)
                    created_files.append(f"{self.temp_dir}/{uid}.stl")
                except Exception as e:
                    print(f"Failed to create mesh for UID: {uid}, Error: {e}")
                    continue
                if socketio:
                    progress = (index + 1) / total_components * 100
                    socketio.emit('progress', {
                        'message': f'Preparing components for graph extraction: {int(progress)}%'
                    }, room=username)

            # Set the threshold for the distance
            inside_threshold = 1e-6  # 0.000001 in scientific notation
            outside_threshold = 2e-2  # 0.02 in scientific notation
            
            # Set the fidelity of the mesh sampling
            fidelity = 500

            if socketio:
                socketio.emit('status', {
                    'message': "Computing distances.."
                }, room=username)
            
            # Compute the edges between the candidate component pairs
            total_pairs = len(bounding_box_pairs)
            total_edges = 0
            for pair_index, (uid1, uid2) in enumerate(bounding_box_pairs):
                
                print("--- checking pair ---")
                print(f"Component 1: ", self.components[uid1][2])
                print(f"Component 2: ", self.components[uid2][2])
                
                mesh1 = meshes.get(uid1)
                mesh2 = meshes.get(uid2)

                if mesh1 is None or mesh2 is None:
                    print(f"Skipping pair ({uid1}, {uid2}) due to missing mesh.")
                    continue

                # Compute actual surface distance
                pts1 = mesh1.sample(fidelity)  # Sample 500 points from the surface
                pts2 = mesh2.sample(fidelity)
            
                
                # Check for minimum distance   
                tree = cKDTree(pts2)
                min_dist = tree.query(pts1)[0].min()
                
                print(f"Minimum distance found: {min_dist}")
                
                # Check for volumetric overlap
                sign, distance, directions = self.check_signed_distance(mesh1, mesh2, pts1, pts2, inside_threshold, outside_threshold, False)
                if sign < 0:
                    print(f"Intersection: Volumetric. Distance: {distance}")
                    self.edges[uid1] = self.edges.get(uid1, []) + [{"uid": uid2, "type": "volumetric", "sign": sign, "distance": distance, "directions": directions}]
                    self.edges[uid2] = self.edges.get(uid2, []) + [{"uid": uid1, "type": "volumetric", "sign": sign, "distance": distance, "directions": self.flip_directions(directions)}]
                
                    total_edges += 1
                elif sign == 0:
                    print(f"Intersection: Surface. Edge added. Distance: {distance}")
                    self.edges[uid1] = self.edges.get(uid1, []) + [{"uid": uid2, "type": "surface", "sign": sign, "distance": distance, "directions": directions}]
                    self.edges[uid2] = self.edges.get(uid2, []) + [{"uid": uid1, "type": "surface", "sign": sign, "distance": distance, "directions": self.flip_directions(directions)}]
                    total_edges += 1
                else:
                    print(f"Signed distance: {sign}. Edge not added. Distance: {distance}")
                    
                # send the progress to socketio
                if socketio:
                    progress = (pair_index + 1) / total_pairs * 100
                    socketio.emit('progress', {
                        'message': f'{int(progress)}%'
                    }, room=username)
                    
                print("  ")
                
            # self.update_component_positions()
            
            print(f"Total edges: {total_edges}")
            
        finally:
            # Always clean up temporary files, even if an exception occurs
            if socketio:
                socketio.emit('status', {
                    'message': "Cleaning up temporary files..."
                }, room=username)
                
            # Delete all created STL files
            for stl_file in created_files:
                try:
                    if os.path.exists(stl_file):
                        os.remove(stl_file)
                        print(f"Deleted STL file: {stl_file}")
                except Exception as e:
                    print(f"Failed to delete STL file {stl_file}, Error: {e}")

    # Function to compute the edges between the components (original method)
    def compute_edges(self, socketio=None):
        """Original method - calls new user-specific method with None username"""
        return self.compute_edges_for_user(socketio, None)

