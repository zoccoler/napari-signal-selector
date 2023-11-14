import numpy as np

def linear_interpolate(data):
    from scipy.interpolate import interp1d
    # Extract x and y coordinates
    x = data[:, 0, 0]
    y = data[:, 0, 1]

    # Define a new index for original and interpolated points
    index = np.arange(data.shape[0])
    new_index = np.linspace(0, index[-1], 2*index[-1] + 1)

    # Create interpolation functions for x and y
    fx = interp1d(index, x, kind='linear')
    fy = interp1d(index, y, kind='linear')

    # Interpolate to get new x and y values
    new_x = fx(new_index)
    new_y = fy(new_index)

    # Combine into the new data array
    return np.column_stack((new_x, new_y)).reshape(-1, 1, 2)

def generate_line_segments_array(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    # interpolate segments to have half segment around dot/coordinate
    points = linear_interpolate(points)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
