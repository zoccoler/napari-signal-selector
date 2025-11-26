import numpy as np
import pandas as pd

from napari_signal_selector._line import FeaturesLineWidget


def test_features_line_widget_creation(make_napari_viewer):
    """Test that FeaturesLineWidget can be created."""
    viewer = make_napari_viewer()
    widget = FeaturesLineWidget(viewer)

    assert widget is not None
    assert hasattr(widget, "axes")
    assert hasattr(widget, "canvas")
    assert hasattr(widget, "_selectors")


def test_features_line_widget_selectors(make_napari_viewer):
    """Test that the widget has the required selectors."""
    viewer = make_napari_viewer()
    widget = FeaturesLineWidget(viewer)

    # Check that all required selectors exist
    assert "x" in widget._selectors
    assert "y" in widget._selectors
    assert "object_id" in widget._selectors


def test_features_line_widget_with_data(make_napari_viewer):
    """Test widget with actual labels layer and features."""
    viewer = make_napari_viewer()
    widget = FeaturesLineWidget(viewer)

    # Create labels with features
    labels = np.zeros((100, 100), dtype=int)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:80] = 2

    features = pd.DataFrame(
        {
            "label": [1, 1, 1, 2, 2, 2],
            "frame": [0, 1, 2, 0, 1, 2],
            "intensity": [10, 15, 20, 8, 12, 16],
        }
    )

    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer

    # Check that selectors are populated with feature keys
    assert widget._selectors["x"].count() > 0
    assert widget._selectors["y"].count() > 0
    assert widget._selectors["object_id"].count() > 0


def test_axis_key_properties(make_napari_viewer):
    """Test x_axis_key, y_axis_key, and object_id_axis_key properties."""
    viewer = make_napari_viewer()
    widget = FeaturesLineWidget(viewer)

    # Create labels with features
    labels = np.ones((50, 50), dtype=int)
    features = pd.DataFrame(
        {"label": [1, 1], "time": [0, 1], "value": [5, 10]}
    )

    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer

    # Test setting and getting axis keys
    widget.x_axis_key = "time"
    assert widget.x_axis_key == "time"

    widget.y_axis_key = "value"
    assert widget.y_axis_key == "value"

    widget.object_id_axis_key = "label"
    assert widget.object_id_axis_key == "label"


def test_ready_to_plot(make_napari_viewer):
    """Test the _ready_to_plot method."""
    viewer = make_napari_viewer()
    widget = FeaturesLineWidget(viewer)

    # Should not be ready without data
    assert widget._ready_to_plot() == False

    # Add valid data
    labels = np.ones((50, 50), dtype=int)
    features = pd.DataFrame({"label": [1], "x_val": [0], "y_val": [5]})

    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer

    widget.x_axis_key = "x_val"
    widget.y_axis_key = "y_val"
    widget.object_id_axis_key = "label"

    layer.data[0, 0] = 0  # at least one pixel of background

    # Now should be ready
    assert widget._ready_to_plot() == True


def test_valid_axis_keys(make_napari_viewer):
    """Test _get_valid_axis_keys method."""
    viewer = make_napari_viewer()
    widget = FeaturesLineWidget(viewer)

    # Empty with no layers
    assert len(widget._get_valid_axis_keys()) == 0

    # Add layer with features
    labels = np.ones((50, 50), dtype=int)
    features = pd.DataFrame({"label": [1], "col1": [0], "col2": [5]})

    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer

    keys = widget._get_valid_axis_keys()
    assert "label" in keys
    assert "col1" in keys
    assert "col2" in keys
