import numpy as np
import pandas as pd
from napari_signal_selector._interactive import InteractiveFeaturesLineWidget, InteractiveLine2D


def test_plotter_widget_creation(make_napari_viewer):
    """Test that the interactive plotter widget can be created."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    assert plotter is not None


def test_plotter_dock_widget(make_napari_viewer):
    """Test that the plotter can be docked in the viewer."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    viewer.window.add_dock_widget(plotter, area='right')
    dws = [key for key in viewer.window._dock_widgets.keys()]
    assert len(dws) == 1


def test_plotter_with_labels_layer(make_napari_viewer):
    """Test plotter with a labels layer containing features."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    # Create sample labels and features
    labels = np.zeros((100, 100), dtype=int)
    labels[20:40, 20:40] = 1
    labels[60:80, 60:80] = 2
    
    features = pd.DataFrame({
        'label': [1, 1, 1, 2, 2, 2],
        'frame': [0, 1, 2, 0, 1, 2],
        'intensity': [10, 15, 20, 8, 12, 16]
    })
    
    layer = viewer.add_labels(labels, features=features)
    viewer.layers.selection.active = layer
    
    assert len(plotter.layers) == 1
    assert plotter.layers[0] == layer


def test_interactive_line2d_creation_and_selection():
    """Test that InteractiveLine2D can be created with custom attributes."""
    
    # Create mock canvas
    mock_canvas = type('MockCanvas', (), {'draw_idle': lambda self: None})()
    
    line = InteractiveLine2D(
        xdata=[0, 1, 2],
        ydata=[0, 1, 0],
        label_from_napari_layer=1,
        color_from_napari_layer=(1, 0, 0, 1),
        selected=False,
        canvas=mock_canvas,
        parent_widget=None
    )
    
    assert line.label_from_napari_layer == 1
    assert line.selected == False
    assert line._parent_widget is None
    assert len(line.annotations) == 3
    assert len(line.predictions) == 3

    # Test selection
    line.selected = True
    assert line.selected == True
    assert line.get_linestyle() == '--'
    
    # Test deselection
    line.selected = False
    assert line.selected == False
    assert line.get_linestyle() == '-'


def test_plotter_toolbar_buttons(make_napari_viewer):
    """Test that toolbar buttons exist and are accessible."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    # Check toolbar buttons
    assert 'select' in plotter.custom_toolbar.buttons
    assert 'span_select' in plotter.custom_toolbar.buttons
    assert 'add_annotation' in plotter.custom_toolbar.buttons
    assert 'delete_annotation' in plotter.custom_toolbar.buttons
    
    # Check visibility toggle buttons
    assert plotter.show_selected_button is not None
    assert plotter.show_annotations_button is not None
    assert plotter.show_predictions_button is not None


def test_signal_class_spinbox(make_napari_viewer):
    """Test signal class spinbox functionality."""
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    assert plotter._signal_class == 0
    
    # Change signal class
    plotter._change_signal_class(5)
    assert plotter._signal_class == 5
    assert plotter.signal_class_color_spinbox.value == 5
