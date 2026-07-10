import open3d as o3d
import numpy as np
import copy

def flip_mesh(input_file, output_file, visualize=False):
    mesh = o3d.io.read_triangle_mesh(input_file)

    identity_matrix = np.eye(4)
    identity_matrix[2, 2] = -1

    flipped_mesh = copy.deepcopy(mesh).transform(identity_matrix)
    flipped_mesh.triangles = o3d.utility.Vector3iVector(np.flip(np.asarray(flipped_mesh.triangles), axis=1))
    flipped_mesh.compute_triangle_normals()
    flipped_mesh.orient_triangles()

    if visualize:
        o3d.visualization.draw_geometries([flipped_mesh])

    o3d.io.write_triangle_mesh(output_file, flipped_mesh)

if __name__ == "__main__":
    flip_mesh("cbct_filename.stl", "flipped_cbct_filename.stl")