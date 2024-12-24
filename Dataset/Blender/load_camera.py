import bpy
import math
import time
import os


def initialize_camera(renderer_type):
    original_collection = bpy.context.view_layer.active_layer_collection
    
    cameras = {}
    
    # Create a new collection if it doesn't exist
    collection_name = "cameras"
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
        print(f"Collection '{collection_name}' created and linked to the scene.")
    else:
        new_collection = bpy.data.collections[collection_name]
        print(f"Collection '{collection_name}' already exists.")
        
    if renderer_type == 'EEVEE':
        for position in ['Front', 'Left', 'Back', 'Right', 'Bottom', 'Top']:
            camera_name = f'{str(position)}Camera'
            # Create a new camera
            camera_data = bpy.data.cameras.new(name=camera_name)
            camera_data.type = 'PERSP'
            camera_data.angle = math.radians(90)  # 90 degrees in radians
            camera_object = bpy.data.objects.new(camera_name, camera_data)
            new_collection.objects.link(camera_object)
            print(f"Camera '{camera_object.name}' created and added to collection '{collection_name}'.")
            cameras[str(position)] = camera_object
    elif renderer_type == 'CYCLES':
        for position in ['Front']:
            camera_name = f'{str(position)}Camera'
            camera_data = bpy.data.cameras.new(name=camera_name)
            camera_data.type = 'PANO'
            camera_data.panorama_type = 'EQUIRECTANGULAR'
            camera_object = bpy.data.objects.new(camera_name, camera_data)
            new_collection.objects.link(camera_object)
            print(f"Camera '{camera_object.name}' created and added to collection '{collection_name}'.")

            cameras[str(position)] = camera_object
    else:
        raise ValueError('Must be EEVEE OR CYCLES')

    bpy.context.view_layer.active_layer_collection = original_collection
        
    return cameras

def set_render(renderer_type, frame_start=1, frame_end=30):

    # Set the frame range for the animation
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end  # 2 seconds at 30 FPS

    if renderer_type == 'CYCLES':
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
        bpy.context.scene.render.resolution_y = 576
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
    else:
        # Set render engine to Eevee for fast rendering
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            
        # Set render quality to material preview level
        bpy.context.scene.eevee.taa_render_samples = 16  # Lower sample count for faster render
        
        # Set output settings for images
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        bpy.context.scene.render.resolution_x = 500
        bpy.context.scene.render.resolution_y = 500

        

def set_camera(cameras, start, end, initial_rotation):
    
    rotations = {
        'Front': (90, 0, 0),
        'Left': (90, 0, 90),
        'Back': (90, 0, 180),
        'Right': (90, 0, 270),
        'Bottom': (0, 0, 0),
        'Top': (180, 0, 0),
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


def render_video(renderer_type, cameras, output_path):
        
    # Measure render time
    start_time = time.time()
        
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        
        bpy.context.scene.frame_set(frame)

        if renderer_type == 'CYCLES':
            for face, camera in cameras.items():
                bpy.context.scene.camera = camera
                bpy.context.scene.render.filepath = os.path.join(output_path, f'video_frame{frame-1}.png')
                bpy.ops.render.render(write_still=True)
        else:
            frame_path = os.path.join(output_path, f'frame{str(frame)}')
            os.makedirs(frame_path, exist_ok=True)
            for face, camera in cameras.items():
                bpy.context.scene.camera = camera
                bpy.context.scene.render.filepath = os.path.join(frame_path, f'{face.lower()}.png')
                bpy.ops.render.render(write_still=True)
        
    end_time = time.time()
    render_duration = end_time - start_time
    
    print(f"Rendering completed and saved to '{output_path}'.")
    print(f"Render time: {render_duration:.2f} seconds")
    
    bpy.ops.outliner.orphans_purge()
