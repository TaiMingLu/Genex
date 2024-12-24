import os
from PIL import Image
from math import pi, sin, cos, atan2, hypot, floor
import numpy as np
from numpy import clip

def pixel_to_xyz(i, j, face, edge_length):
    """
    Convert pixel coordinates of the output image to 3D coordinates.

    Parameters:
    i (int): The x-coordinate of the pixel.
    j (int): The y-coordinate of the pixel.
    face (int): The face of the cube (0-5).
    edge_length (int): The length of each edge of the cube face in pixels.

    Returns:
    tuple: A tuple (x, y, z) representing the 3D coordinates.
    """
    a = 2.0 * i / edge_length
    b = 2.0 * j / edge_length

    x = np.zeros_like(a, dtype=float)
    y = np.zeros_like(a, dtype=float)
    z = np.zeros_like(a, dtype=float)

    mask = face == 0
    x[mask], y[mask], z[mask] = -1.0, 1.0 - a[mask], 3.0 - b[mask]

    mask = face == 1
    x[mask], y[mask], z[mask] = a[mask] - 3.0, -1.0, 3.0 - b[mask]

    mask = face == 2
    x[mask], y[mask], z[mask] = 1.0, a[mask] - 5.0, 3.0 - b[mask]

    mask = face == 3
    x[mask], y[mask], z[mask] = 7.0 - a[mask], 1.0, 3.0 - b[mask]

    mask = face == 4
    x[mask], y[mask], z[mask] = b[mask] - 1.0, a[mask] - 5.0, 1.0

    mask = face == 5
    x[mask], y[mask], z[mask] = 5.0 - b[mask], a[mask] - 5.0, -1.0

    return x, y, z

def save_individual_faces(cube_image, edge_length, output_directory):
    """
    Save the individual cube faces to the specified directory.

    Parameters:
    cube_image (Image): The combined cube map image.
    edge_length (int): The length of each edge of the cube face in pixels.
    output_directory (str): The directory to save the images.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    face_names = ['right', 'left', 'top', 'bottom', 'front', 'back']
    face_coordinates = [
        (edge_length * 3, edge_length, edge_length * 4, edge_length * 2),  # right
        (edge_length, edge_length, edge_length * 2, edge_length * 2),      # left
        (2 * edge_length, 0, 3 * edge_length, edge_length),                # top
        (2 * edge_length, 2 * edge_length, 3 * edge_length, 3 * edge_length),  # bottom
        (2 * edge_length, edge_length, 3 * edge_length, edge_length * 2),  # front
        (0, edge_length, edge_length, edge_length * 2)                     # back
    ]

    for face_name, (x1, y1, x2, y2) in zip(face_names, face_coordinates):
        face_img = cube_image.crop((x1, y1, x2, y2))
        face_img.save(os.path.join(output_directory, f'{face_name}.png'))

def convert_panorama_to_cubemap(input_folder, interpolation=False, show=False):
    """
    Convert an equirectangular panorama image to a cube map.

    Parameters:
    input_folder (str): Path to the folder containing the panorama image.
    interpolation (bool): Whether to use bilinear interpolation for sampling.
    show (bool): Whether to display the output cube map image.
    """
    # Load the input equirectangular image
    panorama_image = Image.open(os.path.join(input_folder, 'transformation.png'))
    output_cube_path = os.path.join(input_folder, 'cube.png')
    output_cube_faces_path = os.path.join(input_folder, 'cube')
    input_size = panorama_image.size
    print("Input Panorama size:", input_size)

    # Create an output image with appropriate dimensions for the cube map
    cube_image = Image.new("RGB", (input_size[0], int(input_size[0] * 3 / 4)), "black")

    input_pixels = np.array(panorama_image)
    cube_pixels = np.zeros((cube_image.size[1], cube_image.size[0], 3), dtype=np.uint8)  # initialize with black
    edge_length = input_size[0] // 4  # Length of each edge in pixels

    # Create coordinate grids
    i, j = np.meshgrid(np.arange(cube_image.size[0]), np.arange(cube_image.size[1]), indexing='xy')
    face_index = i // edge_length
    face_index[j < edge_length] = 4  # top
    face_index[j >= 2 * edge_length] = 5  # bottom
    face_index[(j >= edge_length) & (j < 2 * edge_length)] = face_index[(j >= edge_length) & (j < 2 * edge_length)]

    # Convert pixel coordinates to 3D coordinates
    x, y, z = pixel_to_xyz(i, j, face_index, edge_length)
    theta = np.arctan2(y, x)  # Angle in the xy-plane
    r = np.hypot(x, y)  # Distance from origin in the xy-plane
    phi = np.arctan2(z, r)  # Angle from the z-axis

    # Source image coordinates
    uf = 2.0 * edge_length * (theta + pi) / pi
    vf = 2.0 * edge_length * (pi / 2 - phi) / pi

    if interpolation:
        # Nearest-neighbor sampling
        ui_nn = np.round(uf).astype(int)
        vi_nn = np.round(vf).astype(int)
        
        valid_nn = (ui_nn >= 0) & (ui_nn < input_size[0]) & (vi_nn >= 0) & (vi_nn < input_size[1])
        nn_pixels = np.zeros((cube_image.size[1], cube_image.size[0], 3), dtype=np.uint8)  # initialize with black
        nn_pixels[(face_index >= 0) & valid_nn] = input_pixels[vi_nn[(face_index >= 0) & valid_nn], ui_nn[(face_index >= 0) & valid_nn]]

        # Create a mask for pixels that are pure black in the nearest-neighbor result
        black_mask = (nn_pixels == 0).all(axis=-1)

        # Bilinear interpolation
        ui = np.floor(uf).astype(int)
        vi = np.floor(vf).astype(int)
        u2 = ui + 1
        v2 = vi + 1
        mu = uf - ui
        nu = vf - vi

        # Ensure indices are within bounds
        ui = np.clip(ui, 0, input_size[0] - 1)
        vi = np.clip(vi, 0, input_size[1] - 1)
        u2 = np.clip(u2, 0, input_size[0] - 1)
        v2 = np.clip(v2, 0, input_size[1] - 1)

        # Pixel values of the four corners
        A = input_pixels[vi, ui]
        B = input_pixels[vi, u2]
        C = input_pixels[v2, ui]
        D = input_pixels[v2, u2]

        # Interpolate the RGB values
        R = A[:, :, 0] * (1 - mu) * (1 - nu) + B[:, :, 0] * mu * (1 - nu) + C[:, :, 0] * (1 - mu) * nu + D[:, :, 0] * mu * nu
        G = A[:, :, 1] * (1 - mu) * (1 - nu) + B[:, :, 1] * mu * (1 - nu) + C[:, :, 1] * (1 - mu) * nu + D[:, :, 1] * mu * nu
        B = A[:, :, 2] * (1 - mu) * (1 - nu) + B[:, :, 2] * mu * (1 - nu) + C[:, :, 2] * (1 - mu) * nu + D[:, :, 2] * mu * nu

        interp_pixels = np.stack((R, G, B), axis=-1).astype(np.uint8)
        
        # Copy the interpolated result to final output
        cube_pixels = interp_pixels

        # Ensure pure black pixels from nearest-neighbor result are directly used in the final output
        cube_pixels[black_mask] = nn_pixels[black_mask]
    else:
        # Nearest-neighbor sampling
        ui = np.round(uf).astype(int)
        vi = np.round(vf).astype(int)

        valid = (ui >= 0) & (ui < input_size[0]) & (vi >= 0) & (vi < input_size[1])
        cube_pixels[(face_index >= 0) & valid] = input_pixels[vi[(face_index >= 0) & valid], ui[(face_index >= 0) & valid]]


    # First row: set empty spaces to black
    cube_pixels[0:edge_length, 0:edge_length] = [0, 0, 0]
    cube_pixels[0:edge_length, edge_length:2*edge_length] = [0, 0, 0]
    cube_pixels[0:edge_length, 3*edge_length:] = [0, 0, 0]

    # Third row: set empty spaces to black
    cube_pixels[2*edge_length:3*edge_length, 0:edge_length] = [0, 0, 0]
    cube_pixels[2*edge_length:3*edge_length, edge_length:2*edge_length] = [0, 0, 0]
    cube_pixels[2*edge_length:3*edge_length, 3*edge_length:] = [0, 0, 0]

    # Convert the numpy array back to an image
    cube_image = Image.fromarray(cube_pixels)
    cube_image.save(output_cube_path)
    if show:
        cube_image.show()

    print(f'Saved Cube Map to {output_cube_path}')

    # Save individual cube faces
    save_individual_faces(cube_image, edge_length, output_cube_faces_path)
    print(f'Saved Cube Faces to {output_cube_faces_path}')

    return cube_image

def precompute_rotation_matrix(rx, ry, rz):
    # Convert degrees to radians
    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    rz = np.deg2rad(rz)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def apply_rotation(x, y, z, R):
    return R @ np.array([x, y, z])

def cubemap_to_equirectangular(cubemap_images, output_width, output_height):
    """
    Convert cube map images to a panorama image.

    Parameters:
    - cubemap_images: cube image faces dictionary.
    - output_width: Width of the output equirectangular image.
    - output_height: Height of the output equirectangular image.
    - show: Whether to display the output image.
    """

    # Precompute rotation matrix
    rx, ry, rz = 90, -90, 180  # Rotation parameters
    R = precompute_rotation_matrix(rx, ry, rz)

    # Create meshgrid for pixel coordinates
    x = np.linspace(0, output_width - 1, output_width)
    y = np.linspace(0, output_height - 1, output_height)
    xv, yv = np.meshgrid(x, y)

    # Convert equirectangular coordinates to spherical coordinates
    theta = (xv / output_width) * 2 * np.pi - np.pi
    phi = (yv / output_height) * np.pi - (np.pi / 2)

    # Convert spherical coordinates to Cartesian coordinates
    xs = np.cos(phi) * np.cos(theta)
    ys = np.cos(phi) * np.sin(theta)
    zs = np.sin(phi)

    # Apply precomputed rotation using predefined apply_rotation function
    xs, ys, zs = apply_rotation(xs.flatten(), ys.flatten(), zs.flatten(), R)
    xs = xs.reshape((output_height, output_width))
    ys = ys.reshape((output_height, output_width))
    zs = zs.reshape((output_height, output_width))

    # Determine which face of the cubemap this point maps to
    abs_x, abs_y, abs_z = np.abs(xs), np.abs(ys), np.abs(zs)
    face_indices = np.argmax(np.stack([abs_x, abs_y, abs_z], axis=-1), axis=-1)

    equirectangular = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for face_name, face_image in cubemap_images.items():
        if face_name == 'right':
            mask = (face_indices == 0) & (xs > 0)
            u = (-zs[mask] / abs_x[mask] + 1) / 2
            v = (ys[mask] / abs_x[mask] + 1) / 2
        elif face_name == 'left':
            mask = (face_indices == 0) & (xs < 0)
            u = (zs[mask] / abs_x[mask] + 1) / 2
            v = (ys[mask] / abs_x[mask] + 1) / 2
        elif face_name == 'bottom':
            mask = (face_indices == 1) & (ys > 0)
            u = (xs[mask] / abs_y[mask] + 1) / 2
            v = (-zs[mask] / abs_y[mask] + 1) / 2
        elif face_name == 'top':
            mask = (face_indices == 1) & (ys < 0)
            u = (xs[mask] / abs_y[mask] + 1) / 2
            v = (zs[mask] / abs_y[mask] + 1) / 2
        elif face_name == 'front':
            mask = (face_indices == 2) & (zs > 0)
            u = (xs[mask] / abs_z[mask] + 1) / 2
            v = (ys[mask] / abs_z[mask] + 1) / 2
        elif face_name == 'back':
            mask = (face_indices == 2) & (zs < 0)
            u = (-xs[mask] / abs_z[mask] + 1) / 2
            v = (ys[mask] / abs_z[mask] + 1) / 2

        # Convert the face u, v coordinates to pixel coordinates
        face_height, face_width, _ = face_image.shape
        u_pixel = np.clip((u * face_width).astype(int), 0, face_width - 1)
        v_pixel = np.clip((v * face_height).astype(int), 0, face_height - 1)

        # Ensure mask is correctly shaped and boolean
        mask = mask.astype(bool)

        # Create boolean indices for equirectangular assignment
        masked_yv = yv[mask]
        masked_xv = xv[mask]

        # Ensure the index arrays are integer type
        masked_yv = masked_yv.astype(int)
        masked_xv = masked_xv.astype(int)

        # Get the color from the cubemap face and set it in the equirectangular image
        equirectangular[masked_yv, masked_xv] = face_image[v_pixel, u_pixel]

    # Save the result
    equirectangular_image = Image.fromarray(equirectangular)

    return equirectangular_image