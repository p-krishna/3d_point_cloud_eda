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
print("Center of the point cloud:", center)
# translate the point cloud to the origin
pcd.translate(-center)

# %% Voxel downsampling
voxel_downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
print("Number of points after voxel downsampling:", len(voxel_downsampled_pcd.points))
o3d.visualization.draw_geometries([voxel_downsampled_pcd], window_name="Voxel Downsampled Point Cloud")

#%% Statistical outlier removal
near_neighbors = 4
std_ratio = 1.0

pcd, ind = voxel_downsampled_pcd.remove_statistical_outlier(nb_neighbors=near_neighbors, std_ratio=std_ratio)

outlier_cloud = voxel_downsampled_pcd.select_by_index(ind, invert=True)
outlier_cloud.paint_uniform_color([0, 0, 1])  # Blue color
o3d.visualization.draw_geometries([outlier_cloud], window_name="Statistical Outlier Removal")