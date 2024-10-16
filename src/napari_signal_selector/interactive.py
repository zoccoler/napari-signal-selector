from qtpy.QtWidgets import QWidget
from typing import Optional, Tuple, Any
import napari
import numpy as np
import numpy.typing as npt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from .line import FeaturesLineWidget

from napari_signal_selector.utilities import generate_line_segments_array
from matplotlib.widgets import SpanSelector
from qtpy.QtWidgets import QWidget, QLabel, QHBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from qtpy.QtWidgets import QLabel, QWidget
from nap_plot_tools import CustomToolbarWidget, QtColorSpinBox, CustomToolButton, cat10_mod_cmap_first_transparent
from napari.utils.events import Event

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
    cmap = cat10_mod_cmap_first_transparent
    normalizer = Normalize(vmin=0, vmax=cmap.N - 1)
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
                xdata, ydata, c=self._annotations, cmap=self.cmap, norm=self.normalizer,
                marker='x', s=self._default_marker_size*4, zorder=3)
            segments = generate_line_segments_array(xdata, ydata)
            # Repeat predictions for interpolated segments (except first and last ones)
            predictions_with_interpolation = np.repeat(
                self._predictions, 2)[1:-1]
            # Create line collection for predictions
            self._predictions_linecollection = LineCollection(segments, cmap=self.cmap, norm=self.normalizer,
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
    def annotations_visible(self):
        return self._annotations_scatter.get_visible()

    @annotations_visible.setter
    def annotations_visible(self, value):
        self._annotations_scatter.set_visible(value)
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
    def predictions_visible(self):
        return self._predictions_linecollection.get_visible()

    @predictions_visible.setter
    def predictions_visible(self, value):
        self._predictions_linecollection.set_visible(value)
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
        icon_dir = self._get_path_to_icon()
        self.custom_toolbar.add_custom_button(name='select', tooltip="Enable or disable line selection",
                                              default_icon_path=Path(
                                                  icon_dir / "select.png").__str__(),
                                              callback=self.enable_line_selections,
                                              checkable=True,
                                              checked_icon_path=Path(icon_dir / "select_checked.png").__str__())
        self.custom_toolbar.add_custom_button(name='span_select', tooltip="Enable or disable span selection",
                                              default_icon_path=Path(
                                                  icon_dir / "span_select.png").__str__(),
                                              callback=self.enable_span_selections,
                                              checkable=True,
                                              checked_icon_path=Path(icon_dir / "span_select_checked.png").__str__())
        self.custom_toolbar.add_custom_button(name='add_annotation', tooltip="Add selected lines to current signal class",
                                              default_icon_path=Path(
                                                  icon_dir / "add_annotation.png").__str__(),
                                              callback=self.add_annotation,
                                              checkable=False)
        self.custom_toolbar.add_custom_button(name='delete_annotation', tooltip="Delete selected lines class annotation",
                                              default_icon_path=Path(
                                                  icon_dir / "delete_annotation.png").__str__(),
                                              callback=self.remove_annotation,
                                              checkable=False)
        ## Signal Selection Tools ##
        self.signal_selection_tools_layout = QHBoxLayout()
        self.signal_selection_tools_layout.addWidget(self.custom_toolbar)
        self.signal_selection_tools_layout.addWidget(QLabel('Signal class:'))
        self.signal_selection_tools_layout.addWidget(
            self.signal_class_color_spinbox)
        # Add show/hide selected button
        self.show_selected_button = CustomToolButton(
            default_icon_path=Path(icon_dir / "show_all.png").__str__(),
            checked_icon_path=Path(icon_dir / "show_selected.png").__str__(),
        )
        self.show_selected_button.setToolTip(
            'Show all signals or show only selected signals (including corresponding labels in Labels layer)')
        self.show_selected_button.setIconSize(32)
        self.show_selected_button.toggled.connect(
            self._show_selected_signals)
        self.signal_selection_tools_layout.addWidget(self.show_selected_button)
        # Add show/hide annotations button
        self.show_annotations_button = CustomToolButton(
            default_icon_path=Path(
                icon_dir / "hide_annotations.png").__str__(),
            checked_icon_path=Path(
                icon_dir / "show_annotations.png").__str__(),
        )
        self.show_annotations_button.setToolTip(
            'Show or hide annotations')
        self.show_annotations_button.setIconSize(32)
        self.show_annotations_button.toggled.connect(
            self._show_annotations)
        self.show_annotations_button.setChecked(True)
        self.signal_selection_tools_layout.addWidget(
            self.show_annotations_button)
        # Add show/hide predictions button
        self.show_predictions_button = CustomToolButton(
            default_icon_path=Path(
                icon_dir / "hide_predictions.png").__str__(),
            checked_icon_path=Path(
                icon_dir / "show_predictions.png").__str__(),
        )
        self.show_predictions_button.setToolTip(
            'Show or hide predictions')
        self.show_predictions_button.setIconSize(32)
        self.show_predictions_button.toggled.connect(
            self._show_predictions)
        self.signal_selection_tools_layout.addWidget(
            self.show_predictions_button)

        # Add stretch to the right to push buttons to the left
        self.signal_selection_tools_layout.addStretch(1)
        # Set the left margin to 0 and spacing to 0
        self.signal_selection_tools_layout.setContentsMargins(0, 0, 0, 0)
        self.signal_selection_tools_layout.setSpacing(0)

        self.layout().insertLayout(1, self.signal_selection_tools_layout)
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

        # Connect new callback
        self.viewer.dims.events.current_step.connect(
            self.on_dims_slider_change)

        # Load previous annotations if any
        self.update_line_layout_from_column('Annotations')
        # Ensure theme is applied
        self.setup_napari_theme(None)

    def _replace_custom_toolbar_icons(self):
        if hasattr(self, 'custom_toolbar'):
            icon_dir = self._get_path_to_icon()
            for button_name, button in self.custom_toolbar.buttons.items():
                button.update_icon_path(default_icon_path=Path(
                    icon_dir / f"{button_name}.png").__str__(), checked_icon_path=Path(icon_dir / f"{button_name}_checked.png").__str__())
            self.show_selected_button.update_icon_path(default_icon_path=Path(icon_dir / "show_all.png").__str__(),
                                                       checked_icon_path=Path(icon_dir / "show_selected.png").__str__())
            self.show_annotations_button.update_icon_path(default_icon_path=Path(icon_dir / "hide_annotations.png").__str__(),
                                                          checked_icon_path=Path(icon_dir / "show_annotations.png").__str__())
            self.show_predictions_button.update_icon_path(default_icon_path=Path(icon_dir / "hide_predictions.png").__str__(),
                                                          checked_icon_path=Path(icon_dir / "show_predictions.png").__str__())

    def setup_napari_theme(self, theme_event: Event):
        super().setup_napari_theme(theme_event)
        if hasattr(self, 'vertical_time_line'):
            if self.vertical_time_line is not None:
                self.vertical_time_line.set_color(self.axes_color)
        self._replace_custom_toolbar_icons()

    def _show_selected_signals(self, checked):
        """Show or hide selected signals corresponding labels in Labels layer.

        Parameters
        ----------
        checked : bool
            True if selected signals are to be shown, False otherwise.
        """
        if checked:
            # If 'Show Selected' label from napari labels layer is checked, uncheck it
            # These must be mutually exclusive
            if self.layers[0].show_selected_label == True:
                self.layers[0].show_selected_label = False
            if len(self._selected_lines) == 0:
                self.mask_labels(None)
            else:
                self.mask_labels(
                    labels_to_keep=[l.label_from_napari_layer for l in self._selected_lines])
        else:
            self.mask_labels(None)

    def _show_annotations(self, checked):
        """Show or hide annotations.

        Parameters
        ----------
        checked : bool
            True if annotations are to be shown, False otherwise.
        """
        # Do not show if all comboboxes have the same value
        if (self.x_axis_key == self.y_axis_key and self.x_axis_key == self.object_id_axis_key):
            for line in self._lines:
                line.annotations_visible = False
            return
        if checked:
            for line in self._lines:
                line.annotations_visible = True
        else:
            for line in self._lines:
                line.annotations_visible = False

    def _show_predictions(self, checked):
        """Show or hide predictions.

        Parameters
        ----------
        checked : bool
            True if predictions are to be shown, False otherwise.
        """
        if checked:
            for line in self._lines:
                line.predictions_visible = True
        else:
            for line in self._lines:
                line.predictions_visible = False

    def on_dims_slider_change(self) -> None:
        # TODO: update vertical line over plot (consider multithreading for performance, check details here:
        #  - https://napari.org/dev/guides/threading.html#multithreading-in-napari)
        if self.viewer.dims.ndim > 2:
            # Do not try to plot if no layers are present yet
            if len(self.layers) == 0:
                return
            self.draw()

    def _update_time_line(self):
        current_time_point = self.viewer.dims.current_step[0]
        if self.vertical_time_line is None:
            self.vertical_time_line = self.axes.axvline(
                x=current_time_point, color=self.axes_color, ls='--')
        else:
            self.vertical_time_line.set_xdata([current_time_point])
        self.axes.add_line(self.vertical_time_line)

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
        if event.type == 'show_selected_label':
            if event.show_selected_label == True:
                if self.show_selected_button.isChecked():
                    # If show_selected_button is checked, uncheck it
                    # These must be mutually exclusive
                    self.show_selected_button.setChecked(False)
        self._draw()

    def add_annotation(self):
        """Add selected lines to current signal class.
        """
        if len(self._selected_lines) > 0:
            self._add_selected_lines_to_features_as_new_category()
            self.show_annotations_button.setChecked(True)

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
            # If show selected button is checked, redraw plot to display only selected lines as the user selects them
            if self.show_selected_button.isChecked():
                self.clear()
                # Show corresponding labels in Labels layer if show_selected_button is checked
                self._show_selected_signals(True)
                # Call draw to show only selected lines (it needs to be called to eventually hide non-selected lines)
                self.draw()
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
        if len(self.layers) == 0:
            # No layers to check for features
            return
        if column_name not in self.layers[0].features.keys():
            return
        for line in self._lines:
            label = line.label_from_napari_layer
            feature_table = self.layers[0].features
            # Get the annotation/predictions for the current object_id from table column
            list_of_values = feature_table[feature_table[self.object_id_axis_key]
                                           == label][column_name].values
            if column_name == 'Predictions':
                line.predictions = list_of_values
                self._show_predictions(True)
            elif column_name == 'Annotations':
                line.annotations = list_of_values
                self._show_annotations(True)

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

    def mask_labels(self, labels_to_keep=None):
        """Mask labels in a labels layer.

        Parameters
        ----------
        labels_to_keep : list of int, optional
            List of labels to keep. If None, all labels are displayed (clears previously masked labels). If empty list, no change is applied.
        """
        from napari.utils import colormaps
        # If emtpy list, return (no labels selected)
        if labels_to_keep == []:
            return
        # Get the first layer (labels layer)
        labels_layer = self.layers[0]
        labels_layer_colormap = labels_layer.colormap
        # Guarantee that the colormap has enough colors to represent all the labels
        if len(labels_layer_colormap.colors) <= labels_layer.data.max():
            labels_layer_colormap = colormaps.label_colormap(
                labels_layer.data.max() + 1)
            labels_layer.colormap = labels_layer_colormap
        if labels_to_keep is None:
            # If labels_to_keep is None, show all labels
            labels_layer_colormap.colors[:, -1] = 1
        else:
            labels_to_hide = [int(l) for l in np.arange(
                1, labels_layer.data.max()+1) if l not in labels_to_keep]
            labels_layer_colormap.colors[labels_to_hide, -1] = 0
        labels_layer.colormap = labels_layer_colormap

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
                if hasattr(self, 'show_selected_button') and self.show_selected_button.isChecked() and len(self._selected_lines) > 0 and j not in [l.label_from_napari_layer - 1 for l in self._selected_lines]:
                    continue
                # if self.show_selected_button.isChecked() and j not in [l.label_from_napari_layer for l in self._selected_lines]:
                #     continue
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
            # Check if annotations are to be shown
            if hasattr(self, 'show_annotations_button'):
                if self.show_annotations_button.isChecked():
                    self.update_line_layout_from_column('Annotations')
                    self._show_annotations(True)
            # Check if predictions are to be shown
            if hasattr(self, 'show_predictions_button'):
                if self.show_predictions_button.isChecked():
                    self.update_line_layout_from_column('Predictions')
                    self._show_predictions(True)

            self.axes.set_xlabel(x_axis_name, color=self.axes_color)
            self.axes.set_ylabel(y_axis_name, color=self.axes_color)
            self.axes.autoscale(enable=True, axis='both', tight=True)
            if hasattr(self, 'vertical_time_line'):
                self._update_time_line()
            self.canvas.draw()
