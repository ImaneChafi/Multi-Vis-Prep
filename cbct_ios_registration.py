import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load meshes (adjust paths to your files)
scanner_mesh_path = '/Users/imanechafi/Desktop/PhD/CBCT_segmentation/127302829_shell_occlusion_l.stl' 
cbct_mesh_path = '/Users/imanechafi/Desktop/PhD/CBCT_segmentation/000.dcm_Segmentation.seg1_Lower Teeth.stl'
 
scanner_mesh = pv.read(scanner_mesh_path)
cbct_mesh = pv.read(cbct_mesh_path)

# Scale normalization function (normalize both meshes based on bounding box dimensions)
def normalize_mesh(mesh):
    bounds = np.array(mesh.bounds).reshape(3, 2)
    scale_factors = bounds[:, 1] - bounds[:, 0]
    mesh.points /= scale_factors.max()
    return mesh

# Apply normalization to both meshes
scanner_mesh = normalize_mesh(scanner_mesh)
cbct_mesh = normalize_mesh(cbct_mesh)

# Function to slice the mesh based on geometric regions (approximating the teeth)
def slice_teeth(mesh, n_slices=8, axis='x'):
    slices = []
    bounds = mesh.bounds
    min_bound = bounds[::2]
    max_bound = bounds[1::2]
    
    # Define slicing planes across the given axis
    if axis == 'x':
        slicing_points = np.linspace(min_bound[0], max_bound[0], n_slices+1)
    elif axis == 'y':
        slicing_points = np.linspace(min_bound[1], max_bound[1], n_slices+1)
    elif axis == 'z':
        slicing_points = np.linspace(min_bound[2], max_bound[2], n_slices+1)

    for i in range(n_slices):
        # Slice between slicing points
        p1 = slicing_points[i]
        p2 = slicing_points[i+1]
        
        # Create the slice and threshold along the chosen axis
        if axis == 'x':
            tooth_slice = mesh.clip_box(bounds=((p1, p2), (min_bound[1], max_bound[1]), (min_bound[2], max_bound[2])))
        elif axis == 'y':
            tooth_slice = mesh.clip_box(bounds=((min_bound[0], max_bound[0]), (p1, p2), (min_bound[2], max_bound[2])))
        elif axis == 'z':
            tooth_slice = mesh.clip_box(bounds=((min_bound[0], max_bound[0]), (min_bound[1], max_bound[1]), (p1, p2)))
        
        slices.append(tooth_slice)

    return slices

# Register the meshes by aligning centroids
def register_meshes(mesh1, mesh2):
    # Translate both meshes to the origin (based on centroids)
    centroid1 = mesh1.center_of_mass()
    centroid2 = mesh2.center_of_mass()
    mesh1.translate(-centroid1)
    mesh2.translate(-centroid2)
    
    return mesh1, mesh2

scanner_mesh, cbct_mesh = register_meshes(scanner_mesh, cbct_mesh)

# Slice the meshes into parts representing teeth (you can adjust the n_slices for better results)
scanner_teeth = slice_teeth(scanner_mesh, n_slices=8, axis='x')
cbct_teeth = slice_teeth(cbct_mesh, n_slices=8, axis='x')

# Create a PyVista plotter and add the registered teeth with opacity and color
plotter = pv.Plotter()

for tooth1, tooth2 in zip(scanner_teeth, cbct_teeth):
    plotter.add_mesh(tooth1, color='blue', opacity=0.5)
    plotter.add_mesh(tooth2, color='red', opacity=0.5)

# Show the overlapped teeth visualization
plotter.show()