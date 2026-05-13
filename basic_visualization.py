#%%
import open3d as o3d

# need this environment variable to avoid "GLFW Error: Wayland: The platform does not support setting the window position")
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

#%%
# 1. Load the point cloud data
pcd = o3d.io.read_point_cloud("sample_data/light-point-cloud-of-my-living-room.ply")

#%%
# 2. Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# %%
# need voxel downsampling to make it work with plotly
pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_plotly([pcd_downsampled])