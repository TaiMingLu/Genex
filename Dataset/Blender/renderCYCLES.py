import bpy
from load_scene import *
from load_camera import *
from sample import *
from tqdm import tqdm
import sys

renderer_type = 'CYCLES'
output_folder = "F:/DATASETS/Panorama/CYCLES" if len(sys.argv) <= 1 else f"F:/DATASETS/Panorama/CYCLES{sys.argv[1]}"
iterations = 10000



# skyBackground()
# load_fbx("F:/DATASETS/ASSETS/City/City.fbx")
# Path to your .blend file
# set_hdri_environment("F:/DATASETS/ASSETS/RealCity/drackenstein_quarry_puresky_8k.exr")
# blend_file_path = "F:/DATASETS/ASSETS/RealCity/object/RealCity.gltf"
# load_gltf_file(blend_file_path)
blend_file_path = "F:/DATASETS/ASSETS/BlockSelected.blend"
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

initialize_camera(renderer_type)
set_render(renderer_type)
cameras = initialize_camera(renderer_type)


os.makedirs(output_folder, exist_ok=True)
print(f'Rendering to {output_folder}')


for i in tqdm(range(iterations)):

    iteration_output_path = os.path.join(output_folder, f'iteration{i}')
    if os.path.exists(iteration_output_path):
        print(f'{iteration_output_path} Exists.')
        continue

    # Use the function to get valid start and end positions and the direction
    start, end, direction, start_trials, direction_trials, angle = get_valid_start_end_and_direction(step_size=12)

    if start and end and direction:
        # Print the start position, end position, direction, and trial counts
        print(f"Start position: {start}, End position: {end}")
        print(f"Direction: {direction}")
        print(f"Angle (radians): {angle}")
        print(f"Angle (degrees): {math.degrees(angle)}")
        print(f"Number of trials to find start location: {start_trials}")
        print(f"Number of trials to find direction: {direction_trials}")
        
        rotation = (0, 0, math.degrees(angle) - 90)
        
        
        set_camera(cameras, start, end, rotation)

        os.makedirs(iteration_output_path, exist_ok=True)
        render_video(renderer_type, cameras, iteration_output_path)

    else:
        print("Failed to find a valid start and end position within the given attempts.")
