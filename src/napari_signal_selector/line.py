from typing import List, Optional, Tuple, Any

import matplotlib.colors as mcolor
import napari
import numpy as np
import numpy.typing as npt
from qtpy.QtWidgets import QWidget
from matplotlib.lines import Line2D
from napari_matplotlib.line import FeaturesLineWidget
from napari_matplotlib.util import Interval

# from .utilities import get_nice_colormap
import colorcet as cc

__all__ = ["InteractiveFeaturesLineWidget"]


class InteractiveLine2D(Line2D):
    """InteractiveLine2D class.

    Extends matplotlib.lines.Line2D class to add custom attributes, like selected and annotation.

    Parameters
    ----------
    Line2D : matplotlib.lines.Line2D
        Matplotlib Line2D object.
    """

    def __init__(self, *args, label_from_napari_layer, color_from_napari_layer,
                 selected=False, annotation=0, categorical_color=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.label_from_napari_layer = label_from_napari_layer
        self.color_from_napari_layer = color_from_napari_layer
        self._selected = selected
        self._annotation = annotation
        self._categorical_color = categorical_color

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = value
        if value == True:
            self.set_linestyle('--')
        elif value == False:
            self.set_linestyle('-')
        self.figure.canvas.draw_idle()

    @property
    def annotation(self):
        return self._annotation

    @annotation.setter
    def annotation(self, value):
        self._annotation = value
        if value > 0:
            self.set_linestyle('-')
            self.set_marker('.')
            # colors = get_nice_colormap()
            cmap = cc.glasbey_category10
            cmap.insert(0, [0, 0, 0, 0])
            annotation_color = cmap[value]
            self.set_markerfacecolor(annotation_color)
            self.set_markeredgecolor(annotation_color)
        elif value == 0:
            self.set_linestyle('-')
            self.set_marker('None')
        self.figure.canvas.draw_idle()

    @property
    def categorical_color(self):
        return self._categorical_color

    @categorical_color.setter
    def categorical_color(self, value):
        self._categorical_color = value
        if value is not None:
            # Get custom colormap
            # cmap = get_nice_colormap()
            cmap = cc.glasbey_category10
            cmap.insert(0, [0, 0, 0, 0])
            color = cmap[value]
            self.set_color(color)
        else:
            # Restore original color
            self.set_color(self.color_from_napari_layer)
        self.figure.canvas.draw_idle()


# Update class below
class InteractiveFeaturesLineWidget(FeaturesLineWidget):
    n_layers_input = Interval(1, 1)
    # All layers that have a .features attributes
    input_layer_types = (
        napari.layers.Labels,
    )
    _selected_lines = []
    _lines = []

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(napari_viewer, parent=parent)
        # Create pick event connection id (used by line selector)
        self.pick_event_connection_id = None
        # Create mouse click event connection id (used to clear selections)
        self.mouse_click_event_connection_id = None
        # Enable line selector
        self._enable_line_selector(True)
        # Enable mouse clicks
        self._enable_mouse_clicks(True)
        # TODO: Add a spinbox similar to 'label' in labels layer to display annotation and category color

    def _enable_line_selector(self, active=False):
        """
        Enable or disable making legend pickable.

        This activates a global 'pick_event' for all artists.
        Filter picked artist in `_on_pick` callback function.
        """
        self.line_selection_active = active
        if active:
            if self.pick_event_connection_id is None:
                # `_on_pick` callback function must be implemented
                self.pick_event_connection_id = self.canvas.figure.canvas.mpl_connect(
                    'pick_event', self._on_pick)

    def _enable_mouse_clicks(self, active=False):
        """
        Enable/disable mouse clicks.

        Link mouse clicks to `onclick` callback function
        """
        if active:
            if self.mouse_click_event_connection_id is None:
                self.mouse_click_event_connection_id = self.canvas.figure.canvas.mpl_connect(
                    'button_press_event', self._on_click)
        else:
            if self.mouse_click_event_connection_id is not None:
                print('Warning: disabling mouse clicks event')
                self.canvas.figure.canvas.mpl_disconnect(
                    self.mouse_click_event_connection_id)

    def _on_click(self, event):
        # Right click clears selections
        if event.button == 3:
            self._clear_selections()
        # Left click resets plot colors (in case categoriy color were set)
        elif event.button == 1:
            self.reset_plot_colors()

    def _clear_selections(self):
        # Clear selected lines
        for line in self._selected_lines:
            line.selected = False
        self._selected_lines = []

        # Clear keyboard event connection
        self.viewer.bind_key('s', None, overwrite=True)
        self.viewer.bind_key('a', None, overwrite=True)
        self.viewer.bind_key('d', None, overwrite=True)

        self.canvas.figure.canvas.draw_idle()

    def _on_pick(self, event):
        artist = event.artist
        if isinstance(artist, Line2D):
            line = artist
            if line.selected == True:
                line.selected = False
                # Remove line to selected lines
                if line in self._selected_lines:
                    self._selected_lines.remove(line)

            else:
                line.selected = True
                # Add line to selected lines
                if line not in self._selected_lines:
                    self._selected_lines.append(line)
        # Selected lines list may be obsolete with table annotations

        # TODO: Add buttons to interface instead of keyboard keys
        # Enable keyboard keys if lines selected
        if len(self._selected_lines) > 0:
            # Enable some keyboard keys if lines are selected
            self.viewer.bind_key('s', self._store_selected_lines_to_features_as_same_category, overwrite=True)
            self.viewer.bind_key('a', self._add_selected_lines_to_features_as_new_category, overwrite=True)
            self.viewer.bind_key('d', self._delete_selected_lines_from_features, overwrite=True)

        self.canvas.figure.canvas.draw_idle()

    def _add_selected_lines_to_features_as_new_category(self, viewer):
        if 'Annotations' not in self.layers[0].features.keys():
            self.layers[0].features['Annotations'] = 0
        max_annotations = np.amax(self.layers[0].features['Annotations'])
        for line in self._selected_lines:
            self.layers[0].features.loc[
                self.layers[0].features['label'] == line.label_from_napari_layer, 'Annotations'] = max_annotations + 1
            line.annotation = max_annotations + 1
        self._clear_selections()

    def _store_selected_lines_to_features_as_same_category(self, viewer):
        if 'Annotations' not in self.layers[0].features.keys():
            self.layers[0].features['Annotations'] = 0
        max_annotations = np.amax(self.layers[0].features['Annotations'])
        for line in self._selected_lines:
            if max_annotations == 0:
                annotation_value = 1
            else:
                annotation_value = max_annotations
            self.layers[0].features.loc[self.layers[0].features['label']
                                        == line.label_from_napari_layer, 'Annotations'] = annotation_value
            line.annotation = annotation_value
        self._clear_selections()

    def _delete_selected_lines_from_features(self, viewer):
        if 'Annotations' not in self.layers[0].features.keys():
            self.layers[0].features['Annotations'] = 0
        for line in self._selected_lines:
            self.layers[0].features.loc[self.layers[0].features['label']
                                        == line.label_from_napari_layer, 'Annotations'] = 0
            line.annotation = 0
        self._clear_selections()

    def update_plot_annotations_with_column(self, column_name):
        for line in self._lines:
            label = line.label_from_napari_layer
            feature_table = self.viewer.layers[0].features
            # table = self.viewer.layers.selection.active.features
            # Get the annotation for the current label from table column
            value = feature_table[feature_table['label'] == label][column_name].values[0]
            line.annotation = value
        return

    def update_plot_colors_with_column(self, column_name):
        for line in self._lines:
            label = line.label_from_napari_layer
            feature_table = self.viewer.layers[0].features
            # table = self.viewer.layers.selection.active.features
            # Get the category value for the current label from table column
            value = feature_table[feature_table['label'] == label][column_name].values[0]
            line.categorical_color = value
        return

    def reset_plot_colors(self):
        for line in self._lines:
            line.categorical_color = None
        return

    def reset_plot_annotations(self):
        for line in self._lines:
            line.annotation = 0
        return

    def _get_data(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], str, str]:
        """Get the plot data.

        Returns
        -------
        x: np.ndarray
            The x data to plot. Returns an empty array if nothing to plot.
        y: np.ndarray
            The y data to plot. Returns an empty array if nothing to plot.
        x_axis_name : str
            The title to display on the x axis. Returns
            an empty string if nothing to plot.
        y_axis_name: str
            The title to display on the y axis. Returns
            an empty string if nothing to plot.
        """
        feature_table = self.layers[0].features

        # Sort features by 'label' and x_axis_key
        feature_table = feature_table.sort_values(by=[self.label_axis_key, self.x_axis_key])
        # Get data for each label
        grouped = feature_table.groupby(self.label_axis_key)
        x = np.array([sub_df[self.x_axis_key].values for label, sub_df in grouped]).T.squeeze()
        y = np.array([sub_df[self.y_axis_key].values for label, sub_df in grouped]).T.squeeze()

        x_axis_name = str(self.x_axis_key)
        y_axis_name = str(self.y_axis_key)
        x_axis_name = self.x_axis_key.replace("_", " ")
        y_axis_name = self.y_axis_key.replace("_", " ")

        return x, y, x_axis_name, y_axis_name

    def draw(self) -> None:
        """
        Plot lines for two features from the currently selected layer, grouped by labels.
        """
        if self._ready_to_plot():
            # gets the data and then plots the data
            x, y, x_axis_name, y_axis_name = self._get_data()

            update_lines = True
            if len(self._lines) == 0:
                update_lines = False

            for j, (signal_x, signal_y) in enumerate(zip(x.T, y.T)):
                if self.layers[0].show_selected_label and j != self.layers[0].selected_label - 1:
                    continue
                label_name = self.y_axis_key

                if update_lines:
                    line = self._lines[j]
                    line.set_xdata(signal_x)
                    line.set_ydata(signal_y)
                    self.axes.add_line(line)
                else:
                    # line = Line2D(xdata=signal_x, ydata=signal_y,)
                    line = InteractiveLine2D(
                        xdata=signal_x, ydata=signal_y,
                        label_from_napari_layer=j + 1,
                        color_from_napari_layer=self.layers[0].get_color(j + 1),
                        color=self.layers[0].get_color(j + 1),
                        label=label_name,
                        linestyle='-',
                        picker=True,
                        pickradius=5)
                    self.axes.add_line(line)
                    self._lines += [line]
            self.axes.set_xlabel(x_axis_name)
            self.axes.set_ylabel(y_axis_name)
            self.axes.autoscale(enable=True, axis='both', tight=True)

            self.canvas.draw()
