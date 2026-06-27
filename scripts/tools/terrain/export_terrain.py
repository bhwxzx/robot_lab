"""
Script to export IsaacLab terrain as a heightfield (PNG) for MuJoCo sim2sim.
"""

import argparse
import os
import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Export terrain as a Height Field (PNG) for MuJoCo.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--output_dir", type=str, default=script_dir, help="Directory to save the exported PNG and TXT file."
    )
    parser.add_argument(
        "--resolution", type=float, default=0.02, help="Resolution of the heightfield (meters per pixel)."
    )
    # Add AppLauncher args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch omniverse
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import terrain config after simulation_app is running to avoid Isaac Sim warnings/errors
    from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrains.terrains_cfg import BLIND_ROUGH_AND_STAIRS_TERRAINS_CFG
    from isaaclab.terrains import TerrainGenerator

    print(f"[INFO] Configuring terrain to be a simplified 4x4 mixed grid...")
    cfg = BLIND_ROUGH_AND_STAIRS_TERRAINS_CFG.copy()
    cfg.num_rows = 4
    cfg.num_cols = 4
    cfg.curriculum = False
    cfg.border_width = 0.0  # Remove the huge 20m empty border to speed up raycasting
    cfg.difficulty_range = (1.0, 1.0) # 临时添加：导出最高难度的地形
    
    # Do not use cache to ensure the new parameters take effect immediately
    cfg.use_cache = False

    print(f"[INFO] Generating terrain...")
    terrain = TerrainGenerator(cfg=cfg)
    
    if not hasattr(terrain, "terrain_mesh"):
        print("[ERROR] Terrain generator does not have a 'terrain_mesh' attribute.")
        simulation_app.close()
        return

    mesh = terrain.terrain_mesh
    bounds = mesh.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    
    print(f"[INFO] Mesh Bounding Box:")
    print(f"       X: [{min_x:.3f}, {max_x:.3f}]")
    print(f"       Y: [{min_y:.3f}, {max_y:.3f}]")
    print(f"       Z: [{min_z:.3f}, {max_z:.3f}]")

    resolution = args_cli.resolution
    print(f"[INFO] Rasterizing terrain at {resolution}m resolution...")
    
    # Calculate grid dimensions
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    num_points = xx.size
    
    # Start rays from slightly above the highest point
    ray_start_z = max_z + 1.0
    ray_origins = np.c_[xx.ravel(), yy.ravel(), np.full(num_points, ray_start_z)]
    ray_directions = np.c_[np.zeros(num_points), np.zeros(num_points), np.full(num_points, -1.0)]
    
    print(f"[INFO] Shooting {num_points} rays in batches... (this might take a moment)")
    # Batch raycasting to avoid memory issues and freezing
    batch_size = 50000
    locations_list = []
    index_ray_list = []
    
    for i in range(0, num_points, batch_size):
        end = min(i + batch_size, num_points)
        print(f"       -> Processing rays {i} to {end}...")
        locs, idx_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins[i:end],
            ray_directions=ray_directions[i:end],
            multiple_hits=False
        )
        if len(idx_ray) > 0:
            locations_list.append(locs)
            index_ray_list.append(idx_ray + i)
            
    if len(locations_list) > 0:
        locations = np.vstack(locations_list)
        index_ray = np.concatenate(index_ray_list)
    else:
        locations = np.array([])
        index_ray = np.array([])
    
    # Create the heightmap and fill it with the base minimum height
    heightmap = np.full(num_points, min_z)
    
    # Some rays might miss if the mesh has holes, but for a terrain they usually hit
    if len(index_ray) > 0:
        heightmap[index_ray] = locations[:, 2]
    
    heightmap = heightmap.reshape(xx.shape)
    
    # Normalize to 0-65535 (16-bit)
    actual_min_z = heightmap.min()
    actual_max_z = heightmap.max()
    
    print(f"[INFO] Sampled heights: min={actual_min_z:.4f}, max={actual_max_z:.4f}")
    
    if actual_max_z > actual_min_z:
        normalized_h = (heightmap - actual_min_z) / (actual_max_z - actual_min_z)
        h_uint16 = (normalized_h * 65535).astype(np.uint16)
    else:
        h_uint16 = np.zeros_like(heightmap, dtype=np.uint16)
    
    # Flip vertically for correct MuJoCo image orientation
    # image origin (0,0) is top-left, meshgrid origin is bottom-left
    h_uint16_img = np.flipud(h_uint16)
    
    os.makedirs(args_cli.output_dir, exist_ok=True)
    out_png = os.path.join(args_cli.output_dir, "blind_rough_and_stairs_terrain.png")
    out_txt = os.path.join(args_cli.output_dir, "blind_rough_and_stairs_terrain_info.txt")
    
    print(f"[INFO] Saving PNG to {out_png}")
    img = Image.fromarray(h_uint16_img, mode='I;16')
    img.save(out_png)
    
    # Write the MuJoCo parameters to a txt file
    # size="half_x half_y max_z base_thickness"
    half_x = (max_x - min_x) / 2.0
    half_y = (max_y - min_y) / 2.0
    
    # MuJoCo expects the fourth parameter (base thickness) to be strictly positive.
    # If the lowest point is negative, its absolute value is the base depth.
    base_thickness = abs(actual_min_z) if actual_min_z < 0 else 0.1
    mujoco_size_str = f"{half_x:.3f} {half_y:.3f} {actual_max_z:.3f} {base_thickness:.3f}"
    
    print(f"[INFO] Saving MuJoCo configuration info to {out_txt}")
    with open(out_txt, "w") as f:
        f.write("=== MuJoCo Heightfield Configuration ===\n\n")
        f.write("You can use the following XML snippet to load the generated terrain:\n\n")
        f.write("<asset>\n")
        f.write(f'    <hfield name="terrain_hfield" file="blind_rough_and_stairs_terrain.png" size="{mujoco_size_str}" />\n')
        f.write("</asset>\n\n")
        f.write("<worldbody>\n")
        f.write('    <!-- Optionally, you can add pos="0 0 0" to adjust its placement -->\n')
        f.write('    <geom type="hfield" hfield="terrain_hfield" material="your_material_here" />\n')
        f.write("</worldbody>\n")
        
    print("[INFO] Export complete.")
    simulation_app.close()

if __name__ == "__main__":
    main()
