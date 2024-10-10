from __future__ import annotations

def load_flashing_polygons_data():
    """Generates an image"""
    from skimage.io import imread
    from pandas import read_csv
    from numpy import newaxis
    from pathlib import Path

    DATA_PATH = Path(__file__).parent / "data"
   
    timelapse = imread(DATA_PATH / "synthetic_timelapse.tif")
    labels = imread(DATA_PATH / "synthetic_labels.tif").astype('uint32')
    table = read_csv(DATA_PATH / "table_synthetic_data_with_annotations.csv")
    return [(timelapse, {'name': 'Synthetic Timelapse'}), 
            (labels, {'name': 'Synthetic Labels', 'features': table, 'opacity': 0.4}, 'labels')]
