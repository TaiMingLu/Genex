import bpy

def skyBackground():
    # Clear existing nodes
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create nodes
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    mapping = nodes.new(type='ShaderNodeMapping')
    gradient_tex = nodes.new(type='ShaderNodeTexGradient')
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    light_path = nodes.new(type='ShaderNodeLightPath')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    background = nodes.new(type='ShaderNodeBackground')
    world_output = nodes.new(type='ShaderNodeOutputWorld')

    # Set node positions to mimic your screenshot for better visualization
    tex_coord.location = (0, 0)
    mapping.location = (200, 0)
    gradient_tex.location = (400, 0)
    color_ramp.location = (600, 0)
    light_path.location = (800, 200)
    mix_shader.location = (1000, 0)
    background.location = (1200, 0)
    world_output.location = (1400, 0)

    # Link nodes
    links = world.node_tree.links
    links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], gradient_tex.inputs['Vector'])
    links.new(gradient_tex.outputs['Color'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], mix_shader.inputs[2])
    links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
    links.new(mix_shader.outputs['Shader'], world_output.inputs['Surface'])
    links.new(background.outputs['Background'], mix_shader.inputs[1])

    # Configure nodes
    mapping.inputs['Rotation'].default_value = (0, -1.5708, 0)  # -90 degrees in Y
    color_ramp.color_ramp.elements[0].color = (0.295, 0.434, 1, 1)  # Blue
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)  # White
    color_ramp.color_ramp.elements[1].position = 0.122
    background.inputs['Strength'].default_value = 0.9

def load_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path)


def set_hdri_environment(hdri_image_path):
    """
    Sets an HDRI environment for the current scene.
    
    :param hdri_image_path: Path to the HDRI image file.
    """
    # Set the World to use nodes
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes

    # Clear default nodes
    for node in world_nodes:
        world_nodes.remove(node)

    # Add new nodes for HDRI setup
    background_node = world_nodes.new(type='ShaderNodeBackground')
    environment_texture_node = world_nodes.new(type='ShaderNodeTexEnvironment')
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')

    # Set up the HDRI image
    environment_texture_node.image = bpy.data.images.load(hdri_image_path)

    # Connect the nodes
    bpy.context.scene.world.node_tree.links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
    bpy.context.scene.world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

def load_glb_file(glb_file_path):
    """
    Loads a GLB file into the current scene with error handling.
    
    :param glb_file_path: Path to the GLB file.
    """
    bpy.ops.import_scene.gltf(filepath=glb_file_path, loglevel=50)  # Setting loglevel to suppress warnings


def load_gltf_file(gltf_file_path):
    """
    Loads a GLTF file into the current scene.
    
    :param gltf_file_path: Path to the GLTF (.gltf) file.
    """
    try:
        bpy.ops.import_scene.gltf(filepath=gltf_file_path)
        print(f"Successfully loaded GLTF file: {gltf_file_path}")
    except RuntimeError as e:
        print(f"Failed to load GLTF file: {e}")

