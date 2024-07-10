'''
preprocessing functions
'''
import tifffile as tif
import os, numpy as np

def open_image(filepath):
    """ opens a tif image """
    img = tif.imread(filepath) 
    return img


def tile_image(image, tile_size):
    """Split the image into smaller tiles of given size."""
    img_height, img_width = image.shape[:2]
    tiles = []
    for i in range(0, img_height, tile_size):
        for j in range(0, img_width, tile_size):
            tile = image[i:i + tile_size, j:j + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
    return tiles


def untile_image(tiles, image_shape, tile_size):
    """Reassemble the image from smaller tiles of given size."""
    img_height, img_width = image_shape[:2]
    image = np.zeros(image_shape, dtype=tiles[0].dtype)
    
    idx = 0
    for i in range(0, img_height, tile_size):
        for j in range(0, img_width, tile_size):
            if idx < len(tiles):
                image[i:i + tile_size, j:j + tile_size] = tiles[idx]
                idx += 1
    return image


def save_tiles(tiles, output_dir, prefix='tile'):
    """Save the tiles to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, tile in enumerate(tiles):
        output_path = os.path.join(output_dir, f"{prefix}_{idx}.tif")
        tif.imwrite(output_path, tile)


def untile_image(tiles, image_shape, tile_size):
    """Reassemble the image from smaller tiles of given size."""
    img_height, img_width = image_shape[:2]
    tile_height, tile_width = tile_size[:2]
    image = np.zeros(image_shape, dtype=tiles[0].dtype)
    
    idx = 0
    for i in range(0, img_height, tile_height):
        for j in range(0, img_width, tile_width):
            if idx < len(tiles):
                image[i:i + tile_height, j:j + tile_width] = tiles[idx]
                idx += 1
    return image


# Test code