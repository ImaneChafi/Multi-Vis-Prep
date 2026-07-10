import numpy as np
import open3d as o3d
from easy_mesh_vtk import *

def crop_mesh(mesh, ref_teeth_labels_to_remove, output_path, gingivaLabel=0.0, tol=0.0):
    """
    Save a part of a mesh with the ref labels + gingiva based on the labels of the prep and the two adjacents using compute_point_cloud_distance
    This function is an adaptation of the crop_mesh() method in easy_mesh_vtk.py.
    Modification: saves the cropped mesh in ply format.
    @param mesh: the mesh to crop
    @param ref_teeth_labels: list of floats, the teeth you want ex: (5.0, 6.0, 7.0)
    @param output_path: path to save the cropped mesh, in ply format
    @param gingivaLabel: float, label of gingiva, in most cases equals 0.0
    @param tol: float, distance threshold to select the gingiva
    """
    Part_mesh = Easy_Mesh()

    # create a jaw mesh
    mesh_jaw = Easy_Mesh()
    jaw_idx = np.where(mesh.cell_attributes['Label'] == gingivaLabel)[0]  # extract the target label
    mesh_jaw.cells = mesh.cells[jaw_idx]
    mesh_jaw.update_cell_ids_and_points()
    jaw_cell_center = (mesh.cells[jaw_idx, 0:3] + mesh.cells[jaw_idx, 3:6] + mesh.cells[jaw_idx, 6:9]) / 3.0

    teeth_idx = [idx for idx, e in enumerate(mesh.cell_attributes['Label']) if e not in ref_teeth_labels_to_remove]
    if tol > 0:
        # get the teeth (together)
        teeth_cell_centers = (mesh.cells[teeth_idx, 0:3] + mesh.cells[teeth_idx, 3:6] + mesh.cells[teeth_idx, 6:9]) / 3.0

        # compute the shortest distance
        teeth_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(teeth_cell_centers)))
        gingiva_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.asarray(jaw_cell_center)))
        # For each point in the source point cloud, compute the distance to the target point cloud.
        dists = gingiva_pc.compute_point_cloud_distance(teeth_pc)
        distsNDArray = np.asarray(dists)
        sorted_min_index = sorted(range(len(distsNDArray)), key=lambda k: distsNDArray[k])
        distsNDArray.sort()
        idx_max = np.max(np.where(distsNDArray < tol), axis=1)

        # save the part of mesh
        Part_mesh.cells = mesh_jaw.cells[sorted_min_index[0:idx_max[0]]]
        Part_mesh.cells = np.append(Part_mesh.cells, mesh.cells[teeth_idx], axis=0)
        Part_mesh.update_cell_ids_and_points()
        Part_mesh.cell_attributes['Label'] = np.zeros([len(teeth_idx) + idx_max[0], 1], dtype=np.int32)
        Part_mesh.cell_attributes['Label'][idx_max[0]:] = mesh.cell_attributes['Label'][teeth_idx]
    else:
        Part_mesh.cells = mesh.cells[teeth_idx]
        Part_mesh.update_cell_ids_and_points()
        Part_mesh.cell_attributes['Label'] = mesh.cell_attributes['Label'][teeth_idx]

    Part_mesh.to_ply(output_path)


def remove_gingiva(filename, output_path):
    """
    @param master_filename: str, path to the original master arch
    @return: str, path to the cropped master arch
    """
    mesh = Easy_Mesh(filename)

    crop_mesh(mesh, [0], output_path)
    return output_path

if __name__ == "__main__":
    remove_gingiva("patient_21_l.vtp",
                   "patient_21_l_without_gingiva.ply")