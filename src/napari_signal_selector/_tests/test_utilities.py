import numpy as np

from napari_signal_selector._utilities import (generate_line_segments_array,
                                               linear_interpolate)


def test_linear_interpolate():
    """Test linear interpolation of line segments."""
    # Create simple line data
    data = np.array([[[0, 0]], [[1, 1]], [[2, 2]]])

    result = linear_interpolate(data)

    # Should have double the points minus 1
    expected_length = 2 * len(data) - 1
    assert len(result) == expected_length

    # Check shape
    assert result.shape[1] == 1
    assert result.shape[2] == 2

    # Check that original points are preserved
    assert np.allclose(result[0], data[0])
    assert np.allclose(result[-1], data[-1])


def test_generate_line_segments_array():
    """Test generation of line segments for matplotlib LineCollection."""
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 0, 1])

    segments = generate_line_segments_array(x, y)

    # Should have one segment for each pair of points (with interpolation)
    assert segments.shape[0] > 0
    assert segments.shape[1] == 2  # Each segment has start and end point
    assert segments.shape[2] == 2  # Each point has x and y


def test_generate_line_segments_single_point():
    """Test segment generation with minimal data."""
    x = np.array([0, 1])
    y = np.array([0, 1])

    segments = generate_line_segments_array(x, y)

    # Should still produce valid segments
    assert segments.shape[0] > 0
    assert segments.shape[1] == 2
    assert segments.shape[2] == 2


def test_interpolate_preserves_endpoints():
    """Test that interpolation preserves start and end points."""
    data = np.array([[[0, 5]], [[10, 15]]])

    result = linear_interpolate(data)

    # First and last points should match original
    assert np.allclose(result[0, 0], data[0, 0])
    assert np.allclose(result[-1, 0], data[-1, 0])
