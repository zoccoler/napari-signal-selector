import numpy as np
import pandas as pd
from napari_signal_selector._interactive import InteractiveFeaturesLineWidget


def test_annotation_workflow(make_napari_viewer):
    """Test the complete annotation workflow."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    # Create labels with features
    labels = np.zeros((100, 100), dtype=int)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:80] = 2
    
    features = pd.DataFrame({
        'label': [1, 1, 2, 2],
        'frame': [0, 1, 0, 1],
        'intensity': [10, 15, 8, 12]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    # Set up axes
    plotter.x_axis_key = 'frame'
    plotter.y_axis_key = 'intensity'
    plotter.object_id_axis_key = 'label'
    
    # Simulate selecting a line
    if len(plotter._lines) > 0:
        line = plotter._lines[0]
        line.selected = True
        plotter._selected_lines.append(line)
        
        # Add annotation
        plotter._change_signal_class(2)
        plotter.add_annotation()
        
        # Check that annotation was added
        assert 'Annotations' in layer.features.columns
        assert np.array_equal(layer.features['Annotations'].values, [2, 2, 0, 0])
        

def test_span_selection(make_napari_viewer):
    """Test span selection on lines."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    # Create test data
    labels = np.ones((50, 50), dtype=int)
    labels[0,0] = 0  # at least one pixel of background
    features = pd.DataFrame({
        'label': [1, 1, 1, 1],
        'time': [0, 1, 2, 3],
        'value': [5, 10, 15, 20]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    plotter.x_axis_key = 'time'
    plotter.y_axis_key = 'value'
    plotter.object_id_axis_key = 'label'
    
    # Select a line and add span
    if len(plotter._lines) > 0:
        line = plotter._lines[0]
        line.selected = True
        plotter._selected_lines.append(line)
        
        # Simulate span selection
        plotter._on_span(0.5, 2.5)
        
        # Check that span indices were set
        assert len(line.span_indices) > 0
        assert line.span_indices == [1, 2]


def test_clear_selections(make_napari_viewer):
    """Test clearing line selections."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    labels = np.ones((50, 50), dtype=int)
    labels[0,0] = 0  # at least one pixel of background
    features = pd.DataFrame({
        'label': [1, 1],
        'x': [0, 1],
        'y': [5, 10]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    plotter.x_axis_key = 'x'
    plotter.y_axis_key = 'y'
    plotter.object_id_axis_key = 'label'
    
    # Add line to selected
    if len(plotter._lines) > 0:
        line = plotter._lines[0]
        line.selected = True
        plotter._selected_lines.append(line)
        
        assert len(plotter._selected_lines) == 1
        
        # Clear selections
        plotter._clear_selections()
        assert len(plotter._selected_lines) == 0
        assert line.selected == False


def test_annotations_visibility(make_napari_viewer):
    """Test toggling annotation visibility."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    labels = np.ones((50, 50), dtype=int)
    labels[0,0] = 0  # at least one pixel of background
    features = pd.DataFrame({
        'label': [1, 1],
        'x': [0, 1],
        'y': [5, 10]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    plotter.x_axis_key = 'x'
    plotter.y_axis_key = 'y'
    plotter.object_id_axis_key = 'label'
    
    # Test visibility toggle
    plotter._show_annotations(True)
    # All lines should have annotations visible
    for line in plotter._lines:
        if line._annotations_scatter:
            assert line.annotations_visible == True
    
    plotter._show_annotations(False)
    for line in plotter._lines:
        if line._annotations_scatter:
            assert line.annotations_visible == False


def test_predictions_visibility(make_napari_viewer):
    """Test toggling prediction visibility."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    labels = np.ones((50, 50), dtype=int)
    labels[0,0] = 0  # at least one pixel of background
    features = pd.DataFrame({
        'label': [1, 1],
        'x': [0, 1],
        'y': [5, 10]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    plotter.x_axis_key = 'x'
    plotter.y_axis_key = 'y'
    plotter.object_id_axis_key = 'label'
    
    # Test visibility toggle
    plotter._show_predictions(True)
    for line in plotter._lines:
        if line._predictions_linecollection:
            assert line.predictions_visible == True
    
    plotter._show_predictions(False)
    for line in plotter._lines:
        if line._predictions_linecollection:
            assert line.predictions_visible == False


def test_remove_annotation(make_napari_viewer):
    """Test removing annotations from selected lines."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    labels = np.ones((50, 50), dtype=int)
    labels[0,0] = 0  # at least one pixel of background
    features = pd.DataFrame({
        'label': [1, 1],
        'x': [0, 1],
        'y': [5, 10],
        'Annotations': [1, 1]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    plotter.x_axis_key = 'x'
    plotter.y_axis_key = 'y'
    plotter.object_id_axis_key = 'label'
    
    if len(plotter._lines) > 0:
        line = plotter._lines[0]
        line.selected = True
        plotter._selected_lines.append(line)
        
        # Remove annotation
        plotter.remove_annotation()
        
        # Check annotations are reset to 0
        annotations = layer.features['Annotations'].values
        assert all(annotations == 0)
