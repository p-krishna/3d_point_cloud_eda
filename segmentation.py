#%%
import open3d as o3d
import numpy as np

# need this environment variable to avoid "GLFW Error: Wayland: The platform does not support setting the window position")
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

#%%
# Load the point cloud data
pcd = o3d.io.read_point_cloud("sample_data/light-point-cloud-of-my-living-room.ply")

#%% Preprocessing
# get center of the point cloud
center = pcd.get_center()
print("Original Center of the point cloud:", center)
# translate the point cloud to the origin
pcd.translate(-center)
print("Center after translation:", pcd.get_center())

# %% Voxel downsampling
sample_pcd = pcd.voxel_down_sample(voxel_size=0.01)
print("Number of points after voxel downsampling:", len(sample_pcd.points))
o3d.visualization.draw_geometries([sample_pcd], window_name="Voxel Downsampled Point Cloud")

#%% Statistical outlier removal
near_neighbors = 20
std_ratio = 1.5

cl, ind = sample_pcd.remove_statistical_outlier(nb_neighbors=near_neighbors, std_ratio=std_ratio)

inlier_cloud = sample_pcd.select_by_index(ind)
outlier_cloud = sample_pcd.select_by_index(ind, invert=True)

inlier_cloud.paint_uniform_color([1, 1, 1])
outlier_cloud.paint_uniform_color([0, 0, 0])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#%%
# Estimate normals
inlier_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)
inlier_cloud.normalize_normals()
o3d.visualization.draw_geometries([inlier_cloud], point_show_normal=True) 
# %%
# Segment the largest plane using RANSAC
plane_model, inliers = inlier_cloud.segment_plane(
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=1000
)
# The plane model is in the form ax + by + cz + d = 0
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
print("Plane inliers:", len(inliers))

# Visualize the segmented plane and the remaining points
plane_cloud = inlier_cloud.select_by_index(inliers)
plane_cloud.paint_uniform_color([1.0, 0.0, 0.0])   # red

remaining_cloud = inlier_cloud.select_by_index(inliers, invert=True)
print("Remaining points after plane removal:", len(remaining_cloud.points))
remaining_cloud.paint_uniform_color([0.0, 0.0, 1.0])   # blue

o3d.visualization.draw_geometries([plane_cloud, remaining_cloud], window_name="Segmented Plane and Remaining Points")
# %%

temp_cloud = inlier_cloud
planes = []
plane_inliers_list = []
plane_clouds = []

colors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
]

for i in range(6):
    if len(temp_cloud.points) < 50:
        print(f"Stopping early at plane {i}: too few points left.")
        break

    plane_model, inliers = temp_cloud.segment_plane(
        distance_threshold=0.05,
        ransac_n=3,
        num_iterations=1000
    )

    if len(inliers) < 100:
        print(f"Stopping early at plane {i}: plane too small ({len(inliers)} points).")
        break

    planes.append(plane_model)
    plane_inliers_list.append(inliers)

    plane_cloud = temp_cloud.select_by_index(inliers)
    plane_cloud.paint_uniform_color(colors[i % len(colors)])
    plane_clouds.append(plane_cloud)

    temp_cloud = temp_cloud.select_by_index(inliers, invert=True)

    print(f"Plane {i}: {len(inliers)} points, remaining: {len(temp_cloud.points)}")

o3d.visualization.draw_geometries(plane_clouds, window_name="Segmented Planes")

# %%
# Cluster remaining points with DBSCAN
labels = np.array(
    remaining_cloud.cluster_dbscan(
        eps=0.05,
        min_points=3,
        print_progress=True
    )
)

max_label = labels.max()
print(f"Number of clusters: {max_label + 1}")
print(f"Noise points: {(labels == -1).sum()}")

# Color each cluster
if max_label >= 0:
    colors = plt_colors = np.random.rand(max_label + 1, 3)
    cluster_colors = np.zeros((len(labels), 3))

    for i in range(len(labels)):
        if labels[i] == -1:
            cluster_colors[i] = [0, 0, 0]   # noise = black
        else:
            cluster_colors[i] = colors[labels[i]]

    remaining_cloud.colors = o3d.utility.Vector3dVector(cluster_colors)

#%%
# Visualize
o3d.visualization.draw_geometries(
    [plane_cloud, remaining_cloud],
    window_name="Plane Segmentation + Clustering"
)