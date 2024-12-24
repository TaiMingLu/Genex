import bpy
import json
import os
import mathutils
import math
import time
from sample import *
from tqdm import tqdm

# Path to your dataset
dataset_path = "F:/DATASETS/OmniGibson/dataset"
output_path = "F:/DATASETS/Panorama/car_step_20_frame_50"

# Specify the scene name and JSON filename
# scene_name = "restaurant_brunch"  # Replace with your scene name
# template = 'best'
# Set the render engine to Cycles, which supports GPU rendering
bpy.context.scene.render.engine = 'CYCLES'

# Set the device to GPU
bpy.context.scene.cycles.device = 'GPU'

# Select the compute device type, e.g., CUDA for Nvidia, OPENCL for AMD
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

# Enable all available GPU devices
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type == 'CUDA' or device.type == 'OPTIX' or device.type == 'OPENCL':
        device.use = True

# Set render quality to best
bpy.context.scene.cycles.samples = 1024  # You can adjust this to a higher value if needed
bpy.context.scene.cycles.use_adaptive_sampling = True
bpy.context.scene.cycles.adaptive_threshold = 0.01  # Lower value for higher quality
bpy.context.scene.cycles.max_bounces = 12  # Increase for better quality
bpy.context.scene.cycles.min_bounces = 3
bpy.context.scene.cycles.caustics_reflective = True
bpy.context.scene.cycles.caustics_refractive = True






scenes = os.listdir(os.path.join(dataset_path, 'scenes'))
for scene_name in ['Beechwood_0_int']:
    
    templates = [template[len(scene_name)+1:].split('.')[0] for template in os.listdir(os.path.join(dataset_path, 'scenes', scene_name, 'json'))]
    for template in ['best']:

        # if os.path.exists(os.path.join(output_path, f'{scene_name}_{template}')):
        #     continue

        print(f'Running {scene_name} with {template}')

        output_with = 500
        ouput_height = 500
        frame_start = 1
        frame_end = 24

        iterations = 10000

        #color_strength = random.random(0.5, 0.9)
            
        # Function to load JSON data
        def load_json(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)

        # Function to create a material with textures
        def create_material(name, texture_paths, use_lighting=True, emission_strength=0.7):
            material = bpy.data.materials.new(name)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Clear all nodes
            for node in nodes:
                nodes.remove(node)

            output_node = nodes.new(type='ShaderNodeOutputMaterial')

            if use_lighting:
                # Add nodes for principled BSDF and emission to maintain brightness
                principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                emission_shader = nodes.new(type='ShaderNodeEmission')
                
                # Mix shader to blend between Principled BSDF and Emission
                mix_shader = nodes.new(type='ShaderNodeMixShader')
                mix_factor = nodes.new(type='ShaderNodeValue')
                
                # Set mix factor to control how much color is maintained
                mix_factor.outputs[0].default_value = emission_strength

                links.new(principled_bsdf.outputs['BSDF'], mix_shader.inputs[1])
                links.new(emission_shader.outputs['Emission'], mix_shader.inputs[2])
                links.new(mix_factor.outputs[0], mix_shader.inputs['Fac'])
                links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])
                
                # Set emission strength to a moderate value
                emission_shader.inputs['Strength'].default_value = 1.0

                # Load textures
                for tex_type, tex_path in texture_paths.items():
                    if os.path.exists(tex_path):
                        tex_image = nodes.new(type='ShaderNodeTexImage')
                        tex_image.image = bpy.data.images.load(tex_path)
                        if tex_type == 'albedo':
                            links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Base Color'])
                            links.new(tex_image.outputs['Color'], emission_shader.inputs['Color'])
                        elif tex_type == 'metalness':
                            tex_image.image.colorspace_settings.name = 'Non-Color'
                            links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Metallic'])
                        elif tex_type == 'normal':
                            tex_normal_map = nodes.new(type='ShaderNodeNormalMap')
                            tex_normal_map.inputs['Strength'].default_value = 1.0
                            links.new(tex_image.outputs['Color'], tex_normal_map.inputs['Color'])
                            links.new(tex_normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])
                        elif tex_type == 'opacity':
                            links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Alpha'])
                        elif tex_type == 'roughness':
                            tex_image.image.colorspace_settings.name = 'Non-Color'
                            links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Roughness'])
                    else:
                        print(f"Texture file not found: {tex_path}")
            else:
                # Add nodes for emission shader to display textures without lighting effects
                emission_shader = nodes.new(type='ShaderNodeEmission')
                links.new(emission_shader.outputs['Emission'], output_node.inputs['Surface'])

                # Load textures
                for tex_type, tex_path in texture_paths.items():
                    if os.path.exists(tex_path):
                        tex_image = nodes.new(type='ShaderNodeTexImage')
                        tex_image.image = bpy.data.images.load(tex_path)
                        if tex_type == 'albedo':
                            links.new(tex_image.outputs['Color'], emission_shader.inputs['Color'])
                        else:
                            print(f"Texture type {tex_type} is not used with emission shader.")
                    else:
                        print(f"Texture file not found: {tex_path}")

            return material

        # Function to apply transformations
        def apply_transformations(obj, location, rotation, scale):
            obj.location = location
            obj.rotation_euler = rotation
            obj.scale = scale
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

        # Function to print the hierarchy of the object
        def print_hierarchy(obj, indent=0):
            print(" " * indent + f"{obj.name} ({obj.type})")
            for child in obj.children:
                print_hierarchy(child, indent + 2)
                

        # Function to create a mesh from a USD file
        def create_mesh_from_usd(usd_path, name, model, location, rotation, scale, materials):
            print(usd_path)
            try:
                bpy.ops.wm.usd_import(filepath=usd_path)
                
                # The imported object is the last selected object
                if bpy.context.selected_objects:
                    obj = bpy.data.objects[model]
                    
                    obj.name = name  # Set the object name to the desired name
            
                    # Apply transformations to the main object
                    apply_transformations(obj, location, rotation, scale)

                    # Assign materials to subparts
                    for part_name, texture_paths in materials.items():
                        
                        material_name = f"{part_name}_{model}_material"
                        
                        material = bpy.data.materials.get(material_name)
                        material = material if material else create_material(material_name, texture_paths)
                                        
                        # Find the specific part by name and apply the material
                        for child in obj.children:
                            if part_name in child.name:
                                
                                sub_child = next((sub_child for sub_child in child.children if sub_child.name.startswith("visuals")), None)
                                
                                if sub_child:
                                    sub_child.name = f'visuals'
                                    
                                    if sub_child.data.materials:
                                        sub_child.data.materials[0] = material
                                    else:
                                        sub_child.data.materials.append(material)
                    return obj
                else:
                    print(f"Failed to import object from {usd_path}")
                    return None
            except Exception as e:
                print(f"Error importing USD file {usd_path}: {e}")
                return None

        # Function to load objects from the JSON data
        def load_objects_from_json(json_data):
            objects_info = json_data['objects_info']['init_info']
            object_states = json_data['state']['object_registry']

            for obj_id, info in objects_info.items():
                if obj_id in object_states:
                    state = object_states[obj_id]
                    pos = state['root_link']['pos']
                    ori = state['root_link']['ori']
                    scale = info['args']['scale']

                    # Convert orientation from quaternion to Euler angles
                    quat = mathutils.Quaternion((ori[3], ori[0], ori[1], ori[2]))  # Blender uses (w, x, y, z)
                    rotation = quat.to_euler('XYZ')  # Specify the correct rotation order

                    # Construct the path to the USD file and textures
                    category = info['args']['category']
                    model = info['args']['model']
                    name = info['args']['name']
                    usd_path = os.path.join(dataset_path, 'objects', category, model, 'usd', f"{model}.usd")

                    # Gather all material texture paths
                    materials = {}
                    material_dir = os.path.join(dataset_path, 'objects', category, model, 'usd', 'materials')
                    for filename in os.listdir(material_dir):
                        if filename.startswith(f"{category}-{model}-"):
                            part_name = filename.split('-')[-2]
                            tex_type = filename.split('-')[-1].split('.')[0]
                            if part_name not in materials:
                                materials[part_name] = {}
                            materials[part_name][tex_type] = os.path.join(material_dir, filename)

                    if os.path.exists(usd_path):
                        # Create and place the object in Blender
                        create_mesh_from_usd(usd_path, name, model, pos, rotation, scale, materials)
                    else:
                        print(f"USD file not found: {usd_path}")

        # Function to load a scene
        def load_scene(scene_name, json_filename):
            scene_json_path = os.path.join(dataset_path, 'scenes', scene_name, 'json', json_filename)

            if os.path.exists(scene_json_path):
                # Load JSON data
                json_data = load_json(scene_json_path)
                
                # Clear all existing objects
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete(use_global=False)

                # Load objects from JSON data
                load_objects_from_json(json_data)
                print(f"Objects loaded from {json_filename} and created in the Blender scene.")
            else:
                print(f"Scene JSON file not found: {scene_json_path}")


        def load_camera():
            original_collection = bpy.context.view_layer.active_layer_collection

            cameras = {}

            # Create a new collection if it doesn't exist
            collection_name = "cameras"
            if (collection := bpy.data.collections.get(collection_name)) is None:
                new_collection = bpy.data.collections.new(collection_name)
                bpy.context.scene.collection.children.link(new_collection)
                print(f"Collection '{collection_name}' created and linked to the scene.")
            else:
                new_collection = collection
                print(f"Collection '{collection_name}' already exists.")

            for position in ['Front']:
                camera_name = f'{str(position)}Camera'

                # Create a new camera
                camera_data = bpy.data.cameras.new(name=camera_name)

                # Set camera to panorama mode with equirectangular projection
                camera_data.type = 'PANO'
                camera_data.panorama_type = 'EQUIRECTANGULAR'

                camera_object = bpy.data.objects.new(camera_name, camera_data)

                new_collection.objects.link(camera_object)

                print(f"Camera '{camera_object.name}' created and added to collection '{collection_name}'.")

                cameras[str(position)] = camera_object

            bpy.context.view_layer.active_layer_collection = original_collection

            # Set the render engine to Cycles, which supports GPU rendering
            bpy.context.scene.render.engine = 'CYCLES'

            # Set the device to GPU
            bpy.context.scene.cycles.device = 'GPU'

            # Select the compute device type, e.g., CUDA for Nvidia, OPENCL for AMD
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

            # Enable all available GPU devices
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            for device in bpy.context.preferences.addons['cycles'].preferences.devices:
                device.use = True

            # Set render quality to reasonable settings
            bpy.context.scene.cycles.samples = 512  # Lowered from 2048 for faster rendering
            bpy.context.scene.cycles.use_adaptive_sampling = True
            bpy.context.scene.cycles.adaptive_threshold = 0.05  # Slightly higher for faster rendering
            bpy.context.scene.cycles.max_bounces = 8  # Lowered for better performance
            bpy.context.scene.cycles.min_bounces = 3
            bpy.context.scene.cycles.caustics_reflective = True
            bpy.context.scene.cycles.caustics_refractive = True

            # Set resolution to 1024x512
            bpy.context.scene.render.resolution_x = 1024
            bpy.context.scene.render.resolution_y = 512

            return cameras

        def set_camera(cameras, start, end, initial_rotation):
            
            rotations = {
                'Front': (90, 0, 0),
            }        
            
            for face, camera_object in cameras.items():    
                
                rotation = tuple(map(sum, zip(rotations[face], initial_rotation)))
                camera_object.rotation_euler = tuple(map(math.radians, rotation))

                # Set initial camera position with z-coordinate of 2
                camera_object.location = start
                camera_object.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_start)

                # Set final camera position
                camera_object.location = end  # Adjust this to your desired end location
                camera_object.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_end)  # 2 seconds at 60 FPS

                # Set keyframe interpolation to linear
                for fcurve in camera_object.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = 'LINEAR'

        def render_video(cameras, output_path):
                
            # Measure render time
            start_time = time.time()
                
            for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
                
                # frame_path = os.path.join(output_path, f'frame{str(frame)}')
                # os.makedirs(frame_path, exist_ok=True)
                
                bpy.context.scene.frame_set(frame)
                
                for face, camera in cameras.items():
                
                    bpy.context.scene.camera = camera
                
                    bpy.context.scene.render.filepath = os.path.join(output_path, f'video_frame{frame-1}.png')
                    bpy.ops.render.render(write_still=True)
                
            end_time = time.time()
            render_duration = end_time - start_time
            
            print(f"Rendering completed and saved to '{output_path}'.")
            print(f"Render time: {render_duration:.2f} seconds")
            
            bpy.ops.outliner.orphans_purge()
            
        json_filename = f"{scene_name}_{template}.json"  # Replace with your JSON filename

        # Load the scene
        load_scene(scene_name, json_filename)

        # Set output size to 500x500
        bpy.context.scene.render.resolution_x = output_with
        bpy.context.scene.render.resolution_y = ouput_height

        # Set the frame range for the animation
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end  # 2 seconds at 30 FPS

        cameras = load_camera()
        #set_camera(cameras, start_position, end_position, camera_rotation)

        #render_video(cameras, output_path)
        

        output_folder = f'{scene_name}_{template}'
        scene_output_path = os.path.join(output_path, output_folder)
        os.makedirs(scene_output_path, exist_ok=True)
        print(f'Rendering to {output_path}')

        for i in tqdm(range(iterations)):

            iteration_output_path = os.path.join(scene_output_path, f'iteration{i}')
            if os.path.exists(iteration_output_path):
                print(f'{iteration_output_path} Exists.')
                continue

            # Use the function to get valid start and end positions and the direction
            start, end, direction, start_trials, direction_trials, angle = get_valid_start_end_and_direction()

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
                render_video(cameras, iteration_output_path)

            else:
                print("Failed to find a valid start and end position within the given attempts.")
