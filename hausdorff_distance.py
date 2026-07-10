import pymeshlab

def compute_hausdorff_distance(ios_file, cbct_file):
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(ios_file)
    ms.load_new_mesh(cbct_file)

    metrics = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1)

    print("Hausdorff Distance Metrics (mm):")
    print(f"Minimum: {metrics['min']}")
    print(f"Maximum: {metrics['max']}")
    print(f"Mean: {metrics['mean']}")
    print(f"RMS : {metrics['RMS']}")

if __name__ == "__main__":
    compute_hausdorff_distance( r"ios.stl", r"cbct.stl")