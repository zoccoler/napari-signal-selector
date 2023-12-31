from qtpy.QtWidgets import QWidget
from typing import Optional, Tuple, Any
import napari
import numpy as np
import numpy.typing as npt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from .line import FeaturesLineWidget

from napari_matplotlib.util import Interval
from napari_signal_selector.utilities import generate_line_segments_array
from matplotlib.widgets import SpanSelector
from qtpy.QtWidgets import QWidget, QLabel, QHBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from qtpy.QtWidgets import QLabel, QWidget
from napari_matplotlib.util import Interval
from nap_plot_tools import CustomToolbarWidget, QtColorSpinBox, get_custom_cat10based_cmap_list

__all__ = ["InteractiveFeaturesLineWidget"]
ICON_ROOT = Path(__file__).parent / "icons"


class InteractiveLine2D(Line2D):
    """InteractiveLine2D class.

    Extends matplotlib.lines.Line2D class to add custom attributes, like selected and annotation.

    Parameters
    ----------
    Line2D : matplotlib.lines.Line2D
        Matplotlib Line2D object.
    """
    cmap = get_custom_cat10based_cmap_list()
    mpl_cmap = ListedColormap(cmap)
    normalizer = Normalize(vmin=0, vmax=len(cmap) - 1)
    _default_alpha = 0.7
    _default_marker_size = 4

    def __init__(self, *args, axes=None, canvas=None, label_from_napari_layer, color_from_napari_layer,
                 selected=False, annotations=None, predictions=None, span_indices=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self._axes = axes
        self._canvas = canvas
        self.label_from_napari_layer = label_from_napari_layer
        self.color_from_napari_layer = color_from_napari_layer
        self._selected = selected
        self._annotations = annotations
        if self._annotations is None:
            self._annotations = np.zeros(self.get_xdata().shape).tolist()
        self._predictions = predictions
        if self._predictions is None:
            self._predictions = np.zeros(self.get_xdata().shape).tolist()
        self._span_indices = span_indices
        if self._span_indices is None:
            self._span_indices = []
        if self._axes:
            xdata = self.get_xdata()
            ydata = self.get_ydata()
            # Create scatter for annotations
            self._annotations_scatter = self._axes.scatter(
                xdata, ydata, c=self._annotations, cmap=self.mpl_cmap, norm=self.normalizer,
                marker='x', s=self._default_marker_size*4, zorder=3)
            segments = generate_line_segments_array(xdata, ydata)
            # Repeat predictions for interpolated segments (except first and last ones)
            predictions_with_interpolation = np.repeat(
                self._predictions, 2)[1:-1]
            # Create line collection for predictions
            self._predictions_linecollection = LineCollection(segments, cmap=self.mpl_cmap, norm=self.normalizer,
                                                              zorder=4)
            self._predictions_linecollection.set_array(
                predictions_with_interpolation)
        else:
            self._annotations_scatter = None
            self._predictions_linecollection = None

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = value
        if value == True:
            self.set_linestyle('--')
            self.set_alpha(1)
            self.set_linewidth(2)
        elif value == False:
            self.set_linestyle('-')
            self.set_alpha(self._default_alpha)
            self.set_linewidth(1)
        self._canvas.draw_idle()

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, list_of_values):
        self._annotations = list_of_values
        # Update scatter plot array with annotations (which yield marker colors)
        self._annotations_scatter.set_array(self._annotations)
        self._canvas.draw_idle()

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, list_of_values):
        self._predictions = list_of_values
        # Repeat predictions for interpolated segments (except first and last ones)
        predictions_with_interpolation = np.repeat(self._predictions, 2)[1:-1]
        # Update line collection plot array with predictions
        self._predictions_linecollection.set_array(
            predictions_with_interpolation)
        self._canvas.draw_idle()

    @property
    def span_indices(self):
        return self._span_indices

    @span_indices.setter
    def span_indices(self, list_of_values):
        self._span_indices = list_of_values
        if len(list_of_values) == 0:
            self.set_marker('None')
            self.set_markevery(None)
        else:
            self.set_marker('o')
            self.set_markersize(self._default_marker_size)
            # annotation_color = self.cmap[value]
            # self.set_markeredgecolor(annotation_color)
            self.set_markeredgewidth(1)
            self.set_markevery(list_of_values)
        self._canvas.draw_idle()

    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        if hasattr(self, '_annotations_scatter'):
            if self._annotations_scatter:
                xdata, ydata = self.get_data()
                self._annotations_scatter.set_offsets(list(zip(xdata, ydata)))
        if hasattr(self, '_predictions_linecollection'):
            if self._predictions_linecollection:
                xdata, ydata = self.get_data()
                segments = generate_line_segments_array(xdata, ydata)
                self._predictions_linecollection.set_segments(segments)

    def add_to_axes(self):
        if self._axes:
            self._axes.add_line(self)
            self._axes.add_artist(self._annotations_scatter)
            self._axes.add_collection(self._predictions_linecollection)

class InteractiveFeaturesLineWidget(FeaturesLineWidget):
    """InteractiveFeaturesLineWidget class.

    Extends napari_matplotlib.line.FeaturesLineWidget class to add custom attributes, like selected and annotation.

    Parameters
    ----------
    FeaturesLineWidget : napari_matplotlib.line.FeaturesLineWidget
        napari_matplotlib features line widget.

    Returns
    -------
    napari_matplotlib.line.InteractiveFeaturesLineWidget
        a more interactive version of the napari_matplotlib FeaturesLineWidget.
    """
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
        # Set object name
        self.setObjectName('InteractiveFeaturesLineWidget')

        ### ColorSpinBox ###
        self.signal_class_color_spinbox = QtColorSpinBox()
        self.signal_class_color_spinbox.setToolTip(
            ('signal class number to annotate'))
        # Set callback
        self.signal_class_color_spinbox.connect(
            self._change_signal_class)
        
        ### Custom toolbar ###
        self.custom_toolbar = CustomToolbarWidget(self)
        ### Add toolbuttons to toolbar ###
        self.custom_toolbar.add_custom_button(name='select', tooltip="Enable or disable line selection", default_icon_path=Path(
            ICON_ROOT / "select.png").__str__(), callback=self.enable_line_selections, checkable=True, checked_icon_path=Path(ICON_ROOT / "select_checked.png").__str__())
        self.custom_toolbar.add_custom_button(name='span_select', tooltip="Enable or disable span selection", default_icon_path=Path(
            ICON_ROOT / "span_select.png").__str__(), callback=self.enable_span_selections, checkable=True, checked_icon_path=Path(ICON_ROOT / "span_select_checked.png").__str__())
        self.custom_toolbar.add_custom_button(name='add_annotation', tooltip="Add selected lines to current signal class", default_icon_path=Path(
            ICON_ROOT / "add_annotation.png").__str__(), callback=self.add_annotation, checkable=False)
        self.custom_toolbar.add_custom_button(name='delete_annotation', tooltip="Delete selected lines class annotation", default_icon_path=Path(
            ICON_ROOT / "delete_annotation.png").__str__(), callback=self.remove_annotation, checkable=False)

        ## Signal Selection Tools ##
        self.signal_selection_tools_layout = QHBoxLayout()
        self.signal_selection_tools_layout.addWidget(self.custom_toolbar)
        self.signal_selection_tools_layout.addWidget(QLabel('Signal class:'))
        self.signal_selection_tools_layout.addWidget(
            self.signal_class_color_spinbox)
        # self.signal_selection_tools_layout.addLayout(
        #     self.signal_class_color_spinbox_layout)
        # Add stretch to the right to push buttons to the left
        self.signal_selection_tools_layout.addStretch(1)
        # Set the left margin to 0 (or your desired value)
        # self.signal_selection_tools_layout.setContentsMargins(0, self.signal_selection_tools_layout.contentsMargins().top(), 
        #                                    self.signal_selection_tools_layout.contentsMargins().right(), 
        #                                    self.signal_selection_tools_layout.contentsMargins().bottom())
        self.signal_selection_tools_layout.setContentsMargins(0,0,0,0)

        # Optionally, set spacing if needed
        self.signal_selection_tools_layout.setSpacing(0)
        # # Debug stylesheet
        # self.setStyleSheet("""
        #     QWidget {
        #         background-color: yellow; /* Highlight the background */
        #     }
        #     QHBoxLayout {
        #         border: 2px solid red; /* Red border for layout */
        #     }
        #     CustomToolbarWidget {
        #         background-color: lightblue; /* Blue background for custom toolbar */
        #     }
        # """)

        self.layout().insertLayout(2, self.signal_selection_tools_layout)
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Create pick event connection id (used by line selector)
        self.pick_event_connection_id = None
        # Create mouse click event connection id (used to clear selections)
        self.mouse_click_event_connection_id = None
        # Set initial signal class valus to 0
        self._signal_class = 0
        # Initialize current_time_line
        self.vertical_time_line = None

        # Create horizontal Span Selector
        self.span_selector = SpanSelector(ax=self.axes,
                                          onselect=self._on_span,
                                          direction="horizontal",
                                          useblit=True,
                                          props=dict(
                                              alpha=0.5, facecolor="tab:orange"),
                                          interactive=False,
                                          button=1,
                                          drag_from_anywhere=True)
        self.span_selector.active = False
        # Always enable mouse clicks to clear selections (right button)
        self._enable_mouse_clicks(True)

        # z-step changed in viewer
        # Disconnect draw event on z-slider callback (improves performance)
        current_step_callbacks = self.viewer.dims.events.current_step.callbacks
        draw_callback_tuple = [
            callback for callback in current_step_callbacks if callback[1] == '_draw'][0]
        self.viewer.dims.events.current_step.disconnect(draw_callback_tuple)
        # Connect new callback
        self.viewer.dims.events.current_step.connect(
            self.on_dims_slider_change)

    def on_dims_slider_change(self) -> None:
        pass
        # TODO: update vertical line over plot (consider multithreading for performance, check details here:
        #  - https://napari.org/dev/guides/threading.html#multithreading-in-napari)
        # if self.viewer.dims.ndim > 2:
        #     current_time_point = self.viewer.dims.current_step[0]
        #     if self.vertical_time_line is None:
        #         self.vertical_time_line = self.axes.axvline(x=current_time_point, color='white', ls='--')
        #     else:
        #         self.vertical_time_line.set_xdata(current_time_point)
        #     self.canvas.figure.canvas.draw_idle()

    def on_update_layers(self) -> None:
        """
        Called when the layer selection changes by ``self.update_layers()``.
        """
        super().on_update_layers()
        if len(self.layers) > 0:
            if 'show_selected_label' in self.layers[0].events.emitters.keys():
                self.layers[0].events.show_selected_label.connect(
                    self._show_selected_label)
                self.layers[0].events.selected_label.connect(
                    self._show_selected_label)

    def _show_selected_label(self, event: napari.utils.events.Event) -> None:
        """Redraw plot with selected label.

        Parameters
        ----------
        event : napari.utils.events.Event
            napari event.
        """
        self._draw()

    def add_annotation(self):
        """Add selected lines to current signal class.
        """
        if len(self._selected_lines) > 0:
            self._add_selected_lines_to_features_as_new_category()

    def remove_annotation(self):
        """Remove selected lines from current signal class.
        """
        if len(self._selected_lines) > 0:
            self._remove_selected_lines_from_features()

    def _change_signal_class(self, value):
        """Change signal class and updates color box.

        Parameters
        ----------
        value : int
            New signal class value.
        """
        self._signal_class = value
        self.signal_class_color_spinbox.value = value
        # self.colorBox.update()

    def enable_line_selections(self, checked):
        """Enable or disable line selector.

        If enabled, span selector is disabled.
        """
        # Update toolbar buttons actions
        if checked:
            self._enable_line_selector(True)
            # Disable span selector upon activation of line selector
            self.custom_toolbar.set_button_state('span_select', False)
            self._enable_span_selector(False)
        else:
            self._enable_line_selector(False)

    def _enable_line_selector(self, active=False):
        """
        Enable or disable making line pickable.

        This activates a global 'pick_event' for all artists.
        Filter picked artist in `_on_pick` callback function.
        """
        self.line_selection_active = active
        if active:
            if self.pick_event_connection_id is None:
                self.pick_event_connection_id = self.canvas.figure.canvas.mpl_connect(
                    'pick_event', self._on_pick)
        else:
            # TODO: Lines seem to be still selectable after disabling line selector
            if self.pick_event_connection_id is not None:
                self.canvas.figure.canvas.mpl_disconnect(
                    self.pick_event_connection_id)
                self.pick_event_connection_id = None

    def enable_span_selections(self, checked):
        """Enable or disable span selector.

        If enabled, line selector is disabled.
        """
        if checked:
            self._enable_span_selector(True)
            # Disable line selector upon activation of span selector
            self.custom_toolbar.set_button_state('select', False)
            self._enable_line_selector(False)
        else:
            self._enable_span_selector(False)

    def _enable_span_selector(self, active=False):
        """
        Enable or disable span selector.

        If span selector was created, enable or disable it.
        """
        if self.span_selector is not None:
            self.span_selector.active = active

    def _enable_mouse_clicks(self, active=False):
        """
        Enable/disable mouse clicks.

        Links mouse clicks to `_on_click` callback function.
        """
        if active:
            if self.mouse_click_event_connection_id is None:
                self.mouse_click_event_connection_id = self.canvas.figure.canvas.mpl_connect(
                    'button_press_event', self._on_click)
        else:
            if self.mouse_click_event_connection_id is not None:
                self.canvas.figure.canvas.mpl_disconnect(
                    self.mouse_click_event_connection_id)
                self.mouse_click_event_connection_id = None

    def _on_span(self, xmin, xmax):
        """Update span indices for selected lines.

        Parameters
        ----------
        xmin : int
            Minimum x value of span.
        xmax : int
            Maximum x value of span.
        """
        modifiers = QGuiApplication.keyboardModifiers()
        # If lines were drawn, update span indices of selected lines
        if len(self._lines) > 0:
            selected_lines = [line for line in self._lines if line.selected]
            for line in selected_lines:
                x = line.get_xdata()
                indmin, indmax = np.searchsorted(x, (xmin, xmax))
                indmax = min(len(x) - 1, indmax)

                span_indices = np.arange(indmin, indmax).tolist()
                previous_span_indices = line.span_indices
                # Holding 'SHIFT' preserves previous span indices
                if modifiers == Qt.ShiftModifier:
                    span_indices = previous_span_indices + span_indices
                # Update line span indices
                line.span_indices = span_indices

    def _on_click(self, event):
        """Callback function for mouse clicks.

        - Right click clears selections if select tool is enabled.
        - Right click resets span annotations if span selector is enabled.
        - Left click with select tool enabled and shift key pressed, select all lines.
        - Left click resets plot colors (in case category colors were set).

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event.
        """
        modifiers = QGuiApplication.keyboardModifiers()
        if event.button == 3:
            # Right click clears selections if select tool is enabled
            if self.custom_toolbar.get_button_state('select'):
                self._on_span(0, 0)
                self._clear_selections()
            # Right click resets span annotations if span selector is enabled
            elif self.custom_toolbar.get_button_state('span_select'):
                self._on_span(0, 0)
            else:
                # resets plot colors (in case predictions colors were set)
                self.reset_plot_prediction_colors()

        elif event.button == 1:
            # If left-click with select tool enabled and shift key pressed, select all lines
            if self.custom_toolbar.get_button_state('select'):
                if modifiers == Qt.ShiftModifier:
                    self._select_all_lines()

    def _clear_selections(self):
        """Clear all selected lines.
        """
        for line in self._selected_lines:
            line.selected = False
        self._selected_lines = []
        self.canvas.figure.canvas.draw_idle()

    def _on_pick(self, event):
        """Callback function for artist selection.

        If artist is a line, toggle selection.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            Pick event.
        """
        artist = event.artist
        if isinstance(artist, Line2D):
            line = artist
            if line.selected == True:
                line.selected = False
                # Remove line from selected lines
                if line in self._selected_lines:
                    self._selected_lines.remove(line)

            else:
                line.selected = True
                # Add line to selected lines
                if line not in self._selected_lines:
                    self._selected_lines.append(line)
        self.canvas.figure.canvas.draw_idle()

    def _select_all_lines(self):
        """Select all lines.
        """
        for line in self._lines:
            line.selected = True
            if line not in self._selected_lines:
                self._selected_lines.append(line)

    def _add_selected_lines_to_features_as_new_category(self, viewer=None):
        """Add selected lines to current signal class.

        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            napari viewer instance. This may be needed in case this function is called by keyboard shortcuts (check https://napari.org/stable/howtos/connecting_events.html), by default None.
        """
        # Create Annotations column if not present
        if 'Annotations' not in self.layers[0].features.keys():
            self.layers[0].features['Annotations'] = 0
        for line in self._selected_lines:
            # Get table annotations corresponding to selected line
            table_annotations = self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations']
            # Update table annotations with current signal class (if span selected, update only on span indices)
            if len(line.span_indices) > 0:
                span_mask = np.in1d(np.indices(
                    (len(line.annotations),)), line.span_indices)
                table_annotations[span_mask] = self._signal_class
            else:
                table_annotations[:] = self._signal_class
            # Update features and line annotations
            self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations'] = table_annotations
            line.annotations = table_annotations.values.tolist()

    def _remove_selected_lines_from_features(self, viewer=None):
        """Remove selected lines from current signal class.

        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            napari viewer instance. This may be needed in case this function is called by keyboard shortcuts (check https://napari.org/stable/howtos/connecting_events.html), by default None.
        """
        if 'Annotations' not in self.layers[0].features.keys():
            self.layers[0].features['Annotations'] = 0
        for line in self._selected_lines:
            # Get table annotations corresponding to selected line
            table_annotations = self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations']
            # Update table annotations with current signal class (if span selected, update only on span indices)
            if len(line.span_indices) > 0:
                span_mask = np.in1d(np.indices(
                    (len(line.annotations),)), line.span_indices)
                table_annotations[span_mask] = 0
            else:
                table_annotations[:] = 0
            # Update features and line annotations
            self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations'] = table_annotations
            line.annotations = table_annotations.values.tolist()

    def update_line_layout_from_column(self, column_name='Predictions'):
        """Update line layout (line collection) from a column in the features table.

        Line colors are used to display prediction values.

        Parameters
        ----------
        column_name : str
            Name of the column with results from a classification model.
        """
        for line in self._lines:
            label = line.label_from_napari_layer
            feature_table = self.layers[0].features
            # Get the annotation/predictions for the current object_id from table column
            list_of_values = feature_table[feature_table[self.object_id_axis_key]
                                           == label][column_name].values
            if column_name == 'Predictions':
                line.predictions = list_of_values
            elif column_name == 'Annotations':
                line.annotations = list_of_values

    def reset_plot_prediction_colors(self):
        """Reset plot colors to original colors from napari layer (remove categorical colors).
        """
        for line in self._lines:
            line.predictions = np.zeros(line.get_xdata().shape).tolist()
        return

    def reset_plot_annotations(self):
        """Reset plot annotations to 0 (remove annotations).
        """
        for line in self._lines:
            line.annotations = np.zeros(line.get_xdata().shape).tolist()
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

        # Sort features by object_id and x_axis_key
        feature_table = feature_table.sort_values(
            by=[self.object_id_axis_key, self.x_axis_key])
        # Get data for each object_id (usually label)
        grouped = feature_table.groupby(self.object_id_axis_key)

        x, y = [], []
        for label, sub_df in grouped:
            x.append(sub_df[self.x_axis_key].values)
            y.append(sub_df[self.y_axis_key].values)
        # x = np.array([sub_df[self.x_axis_key].values for label, sub_df in grouped]).T.squeeze(axis=-1)
        # y = np.array([sub_df[self.y_axis_key].values for label, sub_df in grouped]).T.squeeze(axis=-1)

        x_axis_name = str(self.x_axis_key)
        y_axis_name = str(self.y_axis_key)
        x_axis_name = self.x_axis_key.replace("_", " ")
        y_axis_name = self.y_axis_key.replace("_", " ")

        return x, y, x_axis_name, y_axis_name

    def draw(self) -> None:
        """
        Plot lines for two features from the currently selected layer, grouped by object_id.
        """
        if self._ready_to_plot():
            # gets the data and then plots the data
            x, y, x_axis_name, y_axis_name = self._get_data()

            update_lines = False
            if len(self._lines) > 0:  # Check if lines were already created
                # if axes is None, it means axes were cleared, so update lines
                # if axes is the same as current axes, update lines
                if self._lines[0].axes is None or self._lines[0].axes == self.axes:
                    update_lines = True
                # if axes is different from current axes, clear lines because widget was closed
                else:
                    # Clear lines because widget was closed
                    self._lines = []

            for j, (signal_x, signal_y) in enumerate(zip(x, y)):
                if self.layers[0].show_selected_label and j != self.layers[0].selected_label - 1:
                    continue
                label_name = self.y_axis_key

                if update_lines:
                    line = self._lines[j]
                    # Update line axes with current axes (in case axes were cleared when changing selected layer for example)
                    line.axes = self.axes
                    line.set_data(signal_x, signal_y)
                else:
                    line = InteractiveLine2D(
                        xdata=signal_x, ydata=signal_y,
                        label_from_napari_layer=j + 1,
                        color_from_napari_layer=self.layers[0].get_color(
                            j + 1),
                        color=self.layers[0].get_color(j + 1),
                        label=label_name,
                        linestyle='-',
                        picker=True,
                        pickradius=2,
                        alpha=0.7,
                        axes=self.axes,
                        canvas=self.figure.canvas)
                    self._lines += [line]
                # Add (or re-add) every line and scatter to axes (in case axes were cleared)
                line.add_to_axes()
            self.axes.set_xlabel(x_axis_name)
            self.axes.set_ylabel(y_axis_name)
            self.axes.autoscale(enable=True, axis='both', tight=True)
            self.canvas.draw()
