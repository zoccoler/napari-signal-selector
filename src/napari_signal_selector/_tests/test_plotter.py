from napari_signal_selector.interactive import InteractiveFeaturesLineWidget

def test_plotter(make_napari_viewer):
    viewer = make_napari_viewer()
    plotter = InteractiveFeaturesLineWidget(viewer)
    
    # TODO: Add tests for the widget

    viewer.window.add_dock_widget(plotter, area='right')
    dws = [key for key in viewer.window._dock_widgets.keys()]
    assert len(dws) == 1