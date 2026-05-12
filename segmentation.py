#%%
from matplotlib.pylab import std
import open3d as o3d

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
near_neighbors = 4
std_ratio = 1.0

cl, ind = sample_pcd.remove_statistical_outlier(nb_neighbors=near_neighbors, std_ratio=std_ratio)

inlier_cloud = sample_pcd.select_by_index(ind)
outlier_cloud = sample_pcd.select_by_index(ind, invert=True)

inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0, 0, 1])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])