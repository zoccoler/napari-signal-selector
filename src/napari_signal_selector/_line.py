from typing import Any, Dict, List, Optional, Tuple, Union
from cycler import cycler
import os
import napari
import numpy as np
import numpy.typing as npt
from pathlib import Path
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget
from qtpy.QtGui import QIcon
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from napari.utils.events import Event
from matplotlib.figure import Figure

ICON_ROOT = Path(__file__).parent / "icons"
__all__ = ["LineBaseWidget", "FeaturesLineWidget"]


class NapariNavigationToolbar(NavigationToolbar2QT):
    """Custom Toolbar style for Napari."""

    def __init__(self, *args, **kwargs) -> None:  
        super().__init__(*args, **kwargs)  
        # self.setIconSize(
        #     from_napari_css_get_size_of(
        #         "QtViewerPushButton", fallback=(28, 28)
        #     )
        # )

    def _update_buttons_checked(self) -> None:
        """Update toggle tool icons when selected/unselected."""
        super()._update_buttons_checked()  
        icon_dir = self.parentWidget()._get_path_to_icon().__str__()

        # changes pan/zoom icons depending on state (checked or not)
        if "pan" in self._actions:
            if self._actions["pan"].isChecked():
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(icon_dir, "Pan_checked.png"))
                )
            else:
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(icon_dir, "Pan.png"))
                )
        if "zoom" in self._actions:
            if self._actions["zoom"].isChecked():
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(icon_dir, "Zoom_checked.png"))
                )
            else:
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(icon_dir, "Zoom.png"))
                )

class LineBaseWidget(QWidget):
    """
    Base class for widgets that do line plots of two datasets against each other.
    """
    def __init__(self, napari_viewer: napari.viewer.Viewer, parent: Optional[QWidget] = None,
                 ):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_layout_engine("constrained")
        self.toolbar = NapariNavigationToolbar(self.canvas, parent=self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)
        # set minimum layout size
        self.setMinimumSize(400, 400)
        self._setup_callbacks()
        self.layers: list[napari.layers.Layer] = []

        self.add_single_axes()
        self.axes_color = None
        self.axes_bg_color = None
        self.setup_napari_theme(None)
        self.viewer.events.theme.connect(self.setup_napari_theme)

    def setup_napari_theme(self, theme_event: Event):
        if theme_event is None:
            theme = self.viewer.theme
        else:
            theme = theme_event.value
        if theme == 'dark':
            self.axes_color = "white"
            self.axes_bg_color = "#262930"
        elif theme == 'light':
            self.axes_color = "black"
            self.axes_bg_color = "#efebe9"
        
        # changing color of axes background to napari main window color
        self.figure.patch.set_facecolor(self.axes_bg_color)
        # changing color of plot background to napari main window color
        self.axes.set_facecolor(self.axes_bg_color)

        # changing colors of all axes
        self.axes.spines["bottom"].set_color(self.axes_color)
        self.axes.spines["top"].set_color(self.axes_color)
        self.axes.spines["right"].set_color(self.axes_color)
        self.axes.spines["left"].set_color(self.axes_color)
        self.axes.xaxis.label.set_color(self.axes_color)
        self.axes.yaxis.label.set_color(self.axes_color)

        # changing colors of axes ticks
        self.axes.tick_params(axis="x", colors=self.axes_color, labelcolor=self.axes_color)
        self.axes.tick_params(axis="y", colors=self.axes_color, labelcolor=self.axes_color)

        # changing colors of axes labels
        self.axes.xaxis.label.set_color(self.axes_color)
        self.axes.yaxis.label.set_color(self.axes_color)
       
        # replace toolbar icons with dark theme icons
        self._replace_toolbar_icons()
        self.canvas.draw()

    def _get_path_to_icon(self) -> Path:
        """
        Get the icons directory (which is theme-dependent).

        Some icons were modified from
        https://github.com/matplotlib/matplotlib/tree/main/lib/matplotlib/mpl-data/images
        Others were drawn from scratch.
        """

        if self.viewer.theme == "light":
            return ICON_ROOT / "black"
        else:
            return ICON_ROOT / "white"

    def _replace_toolbar_icons(self) -> None:
        """
        Modifies toolbar icons to match the napari theme, and add some tooltips.
        """
        icon_dir = self._get_path_to_icon().__str__()
        for action in self.toolbar.actions():
            text = action.text()
            if text == "Pan":
                action.setToolTip(
                    "Pan/Zoom: Left button pans; Right button zooms; "
                    "Click once to activate; Click again to deactivate"
                )
            if text == "Zoom":
                action.setToolTip(
                    "Zoom to rectangle; Click once to activate; "
                    "Click again to deactivate"
                )
            if len(text) > 0:  # i.e. not a separator item
                icon_path = os.path.join(icon_dir, text + ".png")
                action.setIcon(QIcon(icon_path))

    def _setup_callbacks(self) -> None:
        """
        Sets up callbacks.

        Sets up callbacks for when:
        - Layer selection is changed
        """
        # Layer selection changed in viewer
        self.viewer.layers.selection.events.changed.connect(
            self._update_layers
        )

    @property
    def _valid_layer_selection(self) -> bool:
        """
        Return `True` if layer selection is valid.
        """
        return all(
            isinstance(layer, self.input_layer_types) for layer in self.layers
        )

    @property
    def figure(self) -> Figure:
        """Matplotlib figure."""
        return self.canvas.figure

    def add_single_axes(self) -> None:
        """
        Add a single Axes to the figure.

        The Axes is saved on the ``.axes`` attribute for later access.
        """
        self.axes = self.figure.add_subplot()

    def _update_layers(self, event: napari.utils.events.Event) -> None:
        """
        Update the ``layers`` attribute with currently selected layers and re-draw.
        """
        self.layers = list(self.viewer.layers.selection)
        if len(self.layers) == 0:
            return
        self.layers = sorted(self.layers, key=lambda layer: layer.name)
        self.on_update_layers()
        if self._valid_layer_selection:
            self._draw()

    def _draw(self):
        self.clear()
        if self._valid_layer_selection:
            self.draw()
        self.canvas.draw() 

    def clear(self) -> None:
        """
        Clear the axes.
        """
        self.axes.clear()

    def draw(self) -> None:
        """
        Plot lines for the currently selected layers.
        """
        x, y, x_axis_name, y_axis_name = self._get_data()
        self.axes.plot(x, y)
        self.axes.set_xlabel(x_axis_name, color=self.axes_color)
        self.axes.set_ylabel(y_axis_name, color=self.axes_color)

    def _get_data(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], str, str]:
        """Get the plot data.

        This must be implemented on the subclass.

        Returns
        -------
        data : np.ndarray
            The list containing the line plot data.
        x_axis_name : str
            The label to display on the x axis
        y_axis_name: str
            The label to display on the y axis
        """
        raise NotImplementedError
    
    def setCustomToolbar(self, toolbar):
        layout = self.layout()
        # Remove the current toolbar from the layout
        layout.removeWidget(self.toolbar)
        self.toolbar.deleteLater()  # Delete the old toolbar
        # Add the new custom toolbar to the layout
        layout.insertWidget(1, toolbar)
        self.toolbar = toolbar


class LineWidget(LineBaseWidget):
    """
    Plot pixel values of an Image layer underneath a line from a Shapes layer.
    """

    input_layer_types = (napari.layers.Image,
                         napari.layers.Shapes,)

    def _get_data(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], str, str]:
        """
        Get the plot data.

        Returns
        -------
        x, y : np.ndarray
            x and y values of plot data.
        x_axis_name : str
            The title to display on the x axis
        y_axis_name: str
            The title to display on the y axis
        """
        line_data = self._get_line_data()
        if line_data is None:
            return [], [], "", ""
        image_layers = [layer for layer in self.layers if isinstance(layer, napari.layers.Image)]
        if len(image_layers) == 0:
            return [], [], "", ""
        line_pixel_coords = self._get_line_pixel_coordinates(
            line_data[0], line_data[1], weight=1, shape=image_layers[0].data.shape)

        x = self._get_pixel_distances(line_pixel_coords, line_data[0])
        y = image_layers[0].data[self.current_z][line_pixel_coords[0], line_pixel_coords[1]]
        x_axis_name = 'pixel distance'
        y_axis_name = image_layers[0].name

        return x, y, x_axis_name, y_axis_name

    def _get_line_data(self):
        """
        Get the line data from the Shapes layer.
        """
        for layer in self.layers:
            # There must be a Shapes layer
            if isinstance(layer, napari.layers.Shapes):
                # There must be a line
                if 'line' in layer.shape_type:
                    line_data = layer.data[layer.shape_type.index('line')]
                    return line_data
        return None

    def _get_line_pixel_coordinates(self, start, end, weight=1, shape=None):
        """
        Get the pixel coordinates of a line from start to end using a bezier curve.
        """
        import numpy as np
        from skimage.draw import bezier_curve
        middle = (start + end) / 2
        start = np.round(start).astype(int)
        middle = np.round(middle).astype(int)
        end = np.round(end).astype(int)
        rr, cc = bezier_curve(start[0], start[1], middle[0], middle[1], end[0], end[1], weight=weight, shape=shape)
        return np.array([rr, cc])

    def _get_pixel_distances(self, line_coordinates, start):
        """
        Get the pixel distances from the start of the line.
        """
        distances = np.linalg.norm(line_coordinates - start[:, np.newaxis], axis=0)
        return distances


class FeaturesLineWidget(LineBaseWidget):
    """
    Widget to do line plots of two features from a layer, grouped by object_id.
    """

    # Currently working with Labels layer
    input_layer_types = (
        napari.layers.Labels,
    )

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(napari_viewer, parent=parent)

        self.layout().addLayout(QVBoxLayout())

        self._selectors: Dict[str, QComboBox] = {}
        # Add split-by selector
        self._selectors["object_id"] = QComboBox()
        self._selectors["object_id"].currentTextChanged.connect(self._draw)
        self.layout().addWidget(QLabel(f"object_id:"))
        self.layout().addWidget(self._selectors["object_id"])

        for dim in ["x", "y"]:
            self._selectors[dim] = QComboBox()
            # Re-draw when combo boxes are updated
            self._selectors[dim].currentTextChanged.connect(self._draw)

            self.layout().addWidget(QLabel(f"{dim}-axis:"))
            self.layout().addWidget(self._selectors[dim])

        self._update_layers(None)

    @property
    def x_axis_key(self) -> Union[str, None]:
        """
        Key for the x-axis data.
        """
        if self._selectors["x"].count() == 0:
            return None
        else:
            return self._selectors["x"].currentText()

    @x_axis_key.setter
    def x_axis_key(self, key: str) -> None:
        self._selectors["x"].setCurrentText(key)
        self._draw()

    @property
    def y_axis_key(self) -> Union[str, None]:
        """
        Key for the y-axis data.
        """
        if self._selectors["y"].count() == 0:
            return None
        else:
            return self._selectors["y"].currentText()

    @y_axis_key.setter
    def y_axis_key(self, key: str) -> None:
        self._selectors["y"].setCurrentText(key)
        self._draw()

    @property
    def object_id_axis_key(self) -> Union[str, None]:
        """
        Key for the object_id factor.
        """
        if self._selectors["object_id"].count() == 0:
            return None
        else:
            return self._selectors["object_id"].currentText()

    @object_id_axis_key.setter
    def object_id_axis_key(self, key: str) -> None:
        self._selectors["object_id"].setCurrentText(key)
        self._draw()

    def _get_valid_axis_keys(self) -> List[str]:
        """
        Get the valid axis keys from the layer FeatureTable.

        Returns
        -------
        axis_keys : List[str]
            The valid axis keys in the FeatureTable. If the table is empty
            or there isn't a table, returns an empty list.
        """
        if len(self.layers) == 0 or not (hasattr(self.layers[0], "features")):
            return []
        else:
            return self.layers[0].features.keys()

    def _check_valid_object_id_data_and_set_color_cycle(self):
        # If no features, return False
        if self.layers[0].features is None or len(self.layers[0].features) == 0:
            return False
        # If no object_id_axis_key, return False
        if self.object_id_axis_key is None:
            return False
        feature_table = self.layers[0].features
        # Return True if object_ids from table match labels from layer, otherwise False
        object_ids_from_table = np.unique(feature_table[self.object_id_axis_key].values).astype(int)
        labels_from_layer = np.unique(self.layers[0].data)[1:]  # exclude zero
        if np.array_equal(object_ids_from_table, labels_from_layer):
            # Set color cycle
            self._set_color_cycle(object_ids_from_table.tolist())
            return True
        return False

    def _ready_to_plot(self) -> bool:
        """
        Return True if selected layer has a feature table we can plot with,
        the two columns to be plotted have been selected, and object
        identifier (usually 'labels') in the table.
        """
        if not hasattr(self.layers[0], "features"):
            return False

        feature_table = self.layers[0].features
        valid_keys = self._get_valid_axis_keys()
        valid_object_id_data = self._check_valid_object_id_data_and_set_color_cycle()

        return (
            feature_table is not None
            and len(feature_table) > 0
            and self.x_axis_key in valid_keys
            and self.y_axis_key in valid_keys
            and self.object_id_axis_key in valid_keys
            and valid_object_id_data
        )

    def draw(self) -> None:
        """
        Plot lines for two features from the currently selected layer, grouped by object_id.
        """
        if self._ready_to_plot():
            # draw calls _get_data and then plots the data
            super().draw()

    def _set_color_cycle(self, labels):
        """
        Set the color cycle for the plot from the colors in the Labels layer.
        """
        colors = [self.layers[0].get_color(label) for label in labels]
        napari_labels_cycler = (cycler(color=colors))
        self.axes.set_prop_cycle(napari_labels_cycler)

    def _get_data(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], str, str]:
        """
        Get the plot data from the ``features`` attribute of the first
        selected layer grouped by object_id.

        Returns
        -------
        data : List[np.ndarray]
            List contains X and Y columns from the FeatureTable. Returns
            an empty array if nothing to plot.
        x_axis_name : str
            The title to display on the x axis. Returns
            an empty string if nothing to plot.
        y_axis_name: str
            The title to display on the y axis. Returns
            an empty string if nothing to plot.
        """
        feature_table = self.layers[0].features

        # Sort features by object_id and x_axis_key
        feature_table = feature_table.sort_values(by=[self.object_id_axis_key, self.x_axis_key])
        # Get data for each object_id (usually label)
        grouped = feature_table.groupby(self.object_id_axis_key)
        x = np.array([sub_df[self.x_axis_key].values for label, sub_df in grouped]).T.squeeze()
        y = np.array([sub_df[self.y_axis_key].values for label, sub_df in grouped]).T.squeeze()

        x_axis_name = str(self.x_axis_key)
        y_axis_name = str(self.y_axis_key)

        return x, y, x_axis_name, y_axis_name

    def on_update_layers(self) -> None:
        """
        Called when the layer selection changes by ``self.update_layers()``.
        """
        # Clear combobox
        for dim in ["object_id", "x", "y"]:
            while self._selectors[dim].count() > 0:
                self._selectors[dim].removeItem(0)
            # Add keys for newly selected layer
            self._selectors[dim].addItems(self._get_valid_axis_keys())
