from __future__ import annotations
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"

def load_flashing_polygons_data():
    import zipfile
    import requests
    from skimage.io import imread
    from pandas import read_csv
    from pathlib import Path
    from tqdm import tqdm

    # make data dircetory if it does not exist
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    extracted_folder_path = Path(DATA_PATH / "unzipped_flashing_polygons")
    # If extracted folder does not exist or is empty, download and extract the zip file
    if not extracted_folder_path.exists() or (extracted_folder_path.exists() and not any(extracted_folder_path.iterdir())):
    
        zip_url = 'https://github.com/zoccoler/signal_selector_sample_data/raw/main/flashing_polygons.zip'
        zip_file_path = Path(DATA_PATH / "flashing_polygons.zip")
        # Download the zip file
        response = requests.get(zip_url)

        # Total size in bytes.
        total_size = int(response.headers.get('content-length', 0))
        print(f"Total download size: {total_size/1e6} MBytes")
        print(f"Downloading to {zip_file_path}"	)
        block_size = 1024 
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading zip file")
        with open(zip_file_path, 'wb') as zip_file:
            for block in response.iter_content(block_size):
                progress_bar.update(len(block))
                zip_file.write(block)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print(f"ERROR: Something went wrong with the download. File url: {zip_url}")
    
        # Create the target directory
        extracted_folder_path.mkdir(parents=True, exist_ok=True)
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        
        # Delete the zip file after extraction
        Path(zip_file_path).unlink()

    timelapse = imread(extracted_folder_path / "synthetic_timelapse.tif")
    labels = imread(extracted_folder_path / "synthetic_labels.tif").astype('uint32')
    table = read_csv(extracted_folder_path / "table_synthetic_data_temporal_features.csv")
    return [(timelapse, {'name': 'Flashing Polygons Synthetic Timelapse'}), 
            (labels, {'name': 'Flashing Polygons Synthetic Labels', 'features': table, 'opacity': 0.4}, 'labels')]


def load_blinking_polygons_data():
    import zipfile
    import requests
    from skimage.io import imread
    from pandas import read_csv
    from pathlib import Path
    from tqdm import tqdm

    # make data dircetory if it does not exist
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    extracted_folder_path = Path(DATA_PATH / "unzipped_blinking_polygons")
    # If extracted folder does not exist or is empty, download and extract the zip file
    if not extracted_folder_path.exists() or (extracted_folder_path.exists() and not any(extracted_folder_path.iterdir())):
    
        zip_url = 'https://github.com/zoccoler/signal_selector_sample_data/raw/main/blinking_polygons.zip'
        zip_file_path = Path(DATA_PATH / "blinking_polygons.zip")
        # Download the zip file
        response = requests.get(zip_url)

        # Total size in bytes.
        total_size = int(response.headers.get('content-length', 0))
        print(f"Total download size: {total_size/1e6} MBytes")
        print(f"Downloading to {zip_file_path}"	)
        block_size = 1024 
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading zip file")
        with open(zip_file_path, 'wb') as zip_file:
            for block in response.iter_content(block_size):
                progress_bar.update(len(block))
                zip_file.write(block)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print(f"ERROR: Something went wrong with the download. File url: {zip_url}")
    
        # Create the target directory
        extracted_folder_path.mkdir(parents=True, exist_ok=True)
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        
        # Delete the zip file after extraction
        Path(zip_file_path).unlink()
 
    timelapse = imread(extracted_folder_path / "synthetic_timelapse_sub_signals.tif")
    labels = imread(extracted_folder_path / "synthetic_labels_sub_signals.tif").astype('uint32')
    table = read_csv(extracted_folder_path / "table_synthetic_data_temporal_features_with_annotations_sub_signals.csv")
    return [(timelapse, {'name': 'Blinking Polygons Synthetic Timelapse'}), 
            (labels, {'name': 'Blinking Polygons Synthetic Labels Timelapse', 'features': table, 'opacity': 0.4}, 'labels')]
