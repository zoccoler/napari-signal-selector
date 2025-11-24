try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


from ._sample_data import load_flashing_polygons_data, load_blinking_polygons_data
from ._interactive import InteractiveFeaturesLineWidget

__all__ = (
    "InteractiveFeaturesLineWidget",
    "load_flashing_polygons_data",
    "load_blinking_polygons_data",
)
