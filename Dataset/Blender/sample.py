import bpy
import random
import math
import mathutils

def is_location_valid(location):
    """
    Check if the location is valid by ensuring there is no object within 10 meters above.
    """
    ray_origin = location
    ray_direction = (0, 0, 1)  # Cast the ray upwards
    max_distance = 10.0  # 10 meters above

    # Perform ray casting
    hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
        bpy.context.view_layer.depsgraph, ray_origin, ray_direction, distance=max_distance
    )

    return not hit  # Return True if no object is found within 10 meters above


def is_path_clear(start_location, direction, distance):
    """
    Check if the path is clear by casting rays from the start location in the given direction,
    covering a vertical block of 0.4 units (0.2 up and 0.2 down) and a similar horizontal block.
    """
    # Calculate the end location
    end_location = (
        start_location[0] + direction[0] * distance,
        start_location[1] + direction[1] * distance,
        start_location[2]
    )
    
    # Define step size for grid sampling
    vertical_step = 0.1  # Step size for vertical checking
    horizontal_step = 0.1  # Step size for horizontal checking

    # Create a grid of ray origins within the block
    for dz in [-0.2, 0.0, 0.2]:  # Vertical positions (-0.2 to 0.2)
        for dx in [-0.1, 0.0, 0.1]:  # Horizontal X offsets
            for dy in [-0.1, 0.0, 0.1]:  # Horizontal Y offsets
                ray_origin = (
                    start_location[0] + dx,
                    start_location[1] + dy,
                    start_location[2] + dz
                )
                
                ray_direction = (direction[0], direction[1], 0)  # Ignore Z component

                # Check for a hit with each ray
                hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
                    bpy.context.view_layer.depsgraph, ray_origin, ray_direction, distance=distance
                )
                
                if hit:
                    return False, end_location  # Return False if any ray hits an object

    return True, end_location  # Return True if all paths are clear

# def is_path_clear(start_location, direction, distance):
#     """
#     Check if the path is clear by casting a ray from the start location in the given direction.
#     """
#     end_location = (start_location[0] + direction[0] * distance, 
#                     start_location[1] + direction[1] * distance, 
#                     start_location[2])
#     ray_origin = start_location
#     ray_direction = (direction[0], direction[1], 0)
#     hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
#         bpy.context.view_layer.depsgraph, ray_origin, ray_direction
#     )
#     return not hit, end_location  # True if the path is clear, along with the end location

def sample_valid_location(max_attempts=100):
    """
    Sample a valid location by randomly sampling points and checking validity.
    """

    min_x = -3.51543
    max_x = 3.3263
    min_y = -3.61569
    max_y = 3.40674

    trial_count = 0
    while trial_count < max_attempts:
        trial_count += 1
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = 1.2  # Fixed z-value
        location = (x, y, z)
        
        if is_location_valid(location):
            return location, trial_count
    return None, trial_count

def find_valid_path(start_location, max_attempts=10, step_size=10):
    """
    Try to find a valid path from the start location within the given number of attempts.
    """
    trial_count = 0
    for _ in range(max_attempts):
        trial_count += 1
        angle = random.uniform(0, 2 * math.pi)
        direction = (math.cos(angle), math.sin(angle))
        is_clear, final_position = is_path_clear(start_location, direction, step_size)
        if is_clear:
            return (direction[0], direction[1], 0), final_position, trial_count, angle
    return None, None, trial_count, None

def get_valid_start_end_and_direction(max_start_attempts=100, max_direction_attempts=10, step_size=20):
    """
    Find a valid start location, an end location 2 units away in a random direction,
    and return the direction as a vector.
    If a valid path is not found within the limit, resample the start location.
    """
    total_start_trials = 0
    total_direction_trials = 0
    while total_start_trials < max_start_attempts:
        # Sample a valid location
        valid_location, start_trials = sample_valid_location(max_start_attempts)
        total_start_trials += start_trials
        
        if valid_location is None:
            continue
        
        # Try to find a valid path
        direction, final_position, direction_trials, angle = find_valid_path(valid_location, max_direction_attempts, step_size=step_size)
        total_direction_trials += direction_trials
        
        if direction is not None and final_position is not None:
            return valid_location, final_position, direction, total_start_trials, total_direction_trials, angle
        else:
            print("No valid path found, resampling start location.")
    
    return None, None, None, total_start_trials, total_direction_trials, None

