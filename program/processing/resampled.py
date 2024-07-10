'''
Image processing functions
'''
import numpy as np

def resample_image(image, new_shape):
    """ Resample an image to new dimensions using bilinear interpolation """
    if len(image.shape) == 2:
        # Single-channel image
        src_height, src_width = image.shape
        dst_height, dst_width = new_shape
        resampled_image = np.zeros(new_shape, dtype=np.float32)

        scale_x = src_width / dst_width
        scale_y = src_height / dst_height

        for i in range(dst_height):
            for j in range(dst_width):
                src_x = int(j * scale_x)
                src_y = int(i * scale_y)
                resampled_image[i, j] = image[src_y, src_x]

        return resampled_image

    elif len(image.shape) == 3:
        # Multi-channel image
        src_height, src_width, channels = image.shape
        dst_height, dst_width = new_shape
        resampled_image = np.zeros((dst_height, dst_width, channels), dtype=np.float32)

        for c in range(channels):
            resampled_image[:, :, c] = resample_image(image[:, :, c], new_shape)

        return resampled_image


def remove_invalid_values(data):
  """
      Returns: A new numpy array with invalid values replaced with a specified value.
  """
  valid_data = np.nan_to_num(data, nan=np.median(data))
  valid_data = np.clip(valid_data, None, np.max(valid_data))  
  return valid_data


def min_max_scaler(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data)), [np.min(data), np.max(data)]


def standard_scaler(data):
    return (data - np.mean(data)) / np.std(data)


def inverse_min_max_scaler(scaled_data, min_max_values):
    """Reverse the min-max scaling process."""
    min_val, max_val = min_max_values
    min_val = min_val.to(scaled_data.device)  
    max_val = max_val.to(scaled_data.device)  
    original_data = scaled_data * (max_val - min_val) + min_val
    return original_data



