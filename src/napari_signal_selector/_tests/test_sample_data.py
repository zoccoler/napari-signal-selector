from napari_signal_selector._sample_data import load_flashing_polygons_data

def test_open_sample_data():
    layer_data_tuple = load_flashing_polygons_data()
    assert len(layer_data_tuple) == 2
    # Check that example image shape is correct
    assert layer_data_tuple[0][0].shape == (500, 1, 100, 100)
    # Check that example labels shape is correct
    assert layer_data_tuple[1][0].shape == (100, 100)
    # Check that example labels features (table) shape is correct
    assert layer_data_tuple[1][1]['features'].shape == (15000, 3)
