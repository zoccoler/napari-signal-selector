from typing import Optional, Tuple, Any, Union
import napari
import numpy as np
import numpy.typing as npt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from .line import FeaturesLineWidget
from napari_matplotlib.base import NapariNavigationToolbar
from napari_matplotlib.util import Interval
from napari_signal_selector.utilities import get_custom_cat10based_cmap_list, generate_line_segments_array
from matplotlib.widgets import SpanSelector
from qtpy.QtWidgets import QWidget, QSpinBox, QLabel, QHBoxLayout, QCheckBox, QComboBox, QFrame
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QColor, QPainter

from qtpy.QtWidgets import QLabel, QWidget, QSizePolicy
from qtpy.QtGui import QIcon
import os
from napari_matplotlib.util import Interval, from_napari_css_get_size_of

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
    _default_marker_size = 8

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
                marker='x', s=self._default_marker_size, zorder=3)
            segments = generate_line_segments_array(xdata, ydata)
            # Repeat predictions for interpolated segments (except first and last ones)
            predictions_with_interpolation = np.repeat(self._predictions, 2)[1:-1]
            # Create line collection for predictions
            self._predictions_linecollection = LineCollection(segments, cmap=self.mpl_cmap, norm=self.normalizer,
                                                              zorder=4)
            self._predictions_linecollection.set_array(predictions_with_interpolation)
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
        self._predictions_linecollection.set_array(predictions_with_interpolation)
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


class QtColorBox(QWidget):
    """A widget that shows a square with the current signal class color.
    """
    cmap = get_custom_cat10based_cmap_list()

    def __init__(self) -> None:
        super().__init__()
        # TODO: Check why this may be necessary
        # self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._height = 24
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip(('Selected signal class color'))

        self.color = None

    def paintEvent(self, event):
        """Paint the colorbox.  If no color, display a checkerboard pattern.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        signal_class = self.parent()._signal_class
        if signal_class == 0:
            self.color = None
            for i in range(self._height // 4):
                for j in range(self._height // 4):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.setPen(QColor(230, 230, 230))
                        painter.setBrush(QColor(230, 230, 230))
                    else:
                        painter.setPen(QColor(25, 25, 25))
                        painter.setBrush(QColor(25, 25, 25))
                    painter.drawRect(i * 4, j * 4, 5, 5)
        else:
            color = np.round(255 * np.asarray(self.cmap[signal_class])).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)
            self.color = tuple(color)


class CustomNapariNavigationToolbar(NapariNavigationToolbar):
    """Custom Toolbar style for Napari."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.setIconSize(
            from_napari_css_get_size_of(
                "QtViewerPushButton", fallback=(28, 28)
            )
        )
        self.tb_canvas = self.canvas
        self.tb_parent = kwargs['parent']
        self.tb_coordinates = self.coordinates
        self.extra_button_paths = []

    def _add_new_button(self, text, tooltip_text, icon_image_file_path, callback_name, checkable=False, separator=True):
        """Add a new buttons to the toolbar.

        Parameters
        ----------
        text : str
            the text representing the name of the button
        tooltip_text : str
            the tooltip text exhibited when cursor hovers over button
        icon_image_file_path : str
            path to the "png" file containing the button image
        callback_name : function
            function to be called when button is clicked
        separator: bool
            Whether to add a separator before new button
        checkable: bool
            flag that indicates if button should or not be chackable
        """        
        self.extra_button_paths.append(icon_image_file_path)
        self.toolitems.append((text, tooltip_text, icon_image_file_path, callback_name))
        # Get last widget (A QLabel spacer)
        n_widgets = self.layout().count() # get number of widgets
        myWidget = self.layout().itemAt(n_widgets-1).widget() # get last widget
        # Remove last widget (the spacer)
        self.layout().removeWidget(myWidget)
        myWidget.deleteLater()
        if separator:
            # Add a separator
            self.addSeparator()
        # Add custom button (addAction from QToolBar)
        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QToolBar.html#PySide2.QtWidgets.PySide2.QtWidgets.QToolBar.addAction
        a = self.addAction(QIcon(icon_image_file_path), text, getattr(self.tb_parent, callback_name))
        self._actions[text] = a
        if checkable:
            a.setCheckable(True)
        if tooltip_text is not None:
            a.setToolTip(tooltip_text)
       
        ## Rebuild spacer at the very end of toolbar (re-create 'locLabel' created by __init__ from NavigationToolbar2QT)
        # https://github.com/matplotlib/matplotlib/blob/85d7bb370186f2fa86df6ecc3d5cd064eb7f0b45/lib/matplotlib/backends/backend_qt.py#L631
        if self.tb_coordinates:
            self.locLabel = QLabel("", self)
            self.locLabel.setAlignment(
                Qt.AlignRight | Qt.AlignVCenter)
            self.locLabel.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Ignored,
            )
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)      

    def _update_buttons_checked(self) -> None:
        """Update toggle tool icons when selected/unselected."""
        super()._update_buttons_checked()
        icon_dir = self.parentWidget()._get_path_to_icon()

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
        # If new button added and checkable, update state and icon
        if len(self.extra_button_paths)>0:
            for p in self.extra_button_paths:
                path = Path(p)
                extra_button_name = path.stem
                extra_button_dir = path.parent
                if extra_button_name in self._actions:
                    if self._actions[extra_button_name].isChecked():
                        # Button was checked, update icon to checked
                        self._actions[extra_button_name].setIcon(
                            QIcon(os.path.join(extra_button_dir, extra_button_name + "_checked.png"))
                            )
                        self._actions[extra_button_name].setChecked(True)
                        
                    else:
                        # Button unchecked
                        self._actions[extra_button_name].setIcon(
                            QIcon(os.path.join(extra_button_dir, extra_button_name + ".png"))
                            )
                        self._actions[extra_button_name].setChecked(False)


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
        # Create signal class Label text
        label = QLabel('Signal class:')

        # Create signal class color spinbox
        self.colorBox = QtColorBox()
        signal_class_sb = QSpinBox()
        self.signal_class_SpinBox = signal_class_sb
        self.signal_class_SpinBox.setToolTip(('signal class number to annotate'))
        self.signal_class_SpinBox.setMinimum(0)
        self.signal_class_SpinBox.setSingleStep(1)
        self.signal_class_SpinBox.valueChanged.connect(self._change_signal_class)
        # Limit max width of spinbox
        self.signal_class_SpinBox.setMaximumWidth(50)
        self.signal_class_color_spinbox = QHBoxLayout()
        self.signal_class_color_spinbox.addWidget(self.colorBox)
        self.signal_class_color_spinbox.addWidget(self.signal_class_SpinBox)
        # Add stretch to the right to push buttons to the left
        self.signal_class_color_spinbox.addStretch(1)

        # Create and add a vertical line (separator) to the layout
        vline = QFrame(self)
        vline.setLineWidth(2)
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)

        # Create show overlay layout
        self.show_overlay_layout = QHBoxLayout(self)
        # Create and add the checkbox to the layout
        self.show_overlay_checkbox = QCheckBox("Show Overlay", self)
        self.show_overlay_checkbox.stateChanged.connect(self.on_overlay_checkbox_change)
        self.show_overlay_layout.addWidget(self.show_overlay_checkbox)
        # Add overlay selector
        self._selectors["overlay"] = QComboBox()
        self._selectors["overlay"].setToolTip(('Select column with annotations/predictions'))
        self._selectors["overlay"].setVisible(False)
        self._selectors["overlay"].currentTextChanged.connect(self._draw)

        self.show_overlay_layout.addWidget(self._selectors["overlay"])
        # Add stretch to the right to push buttons to the left
        self.show_overlay_layout.addStretch(1)

        signal_selection_box = QHBoxLayout()
        signal_selection_box.addWidget(label)
        signal_selection_box.addLayout(self.signal_class_color_spinbox)
        # Add separator
        signal_selection_box.addWidget(vline)
        # Add show overlay box to signal selection box
        signal_selection_box.addLayout(self.show_overlay_layout)
        signal_selection_box.addStretch(1)
        self.layout().insertLayout(2, signal_selection_box)

        # Create an instance of your custom toolbar
        custom_toolbar = CustomNapariNavigationToolbar(self.canvas, parent=self)
        # Replace the default toolbar with the custom one
        self.setCustomToolbar(custom_toolbar)
        self._replace_toolbar_icons()

        # Add span selection button to toolbar
        select_icon_file_path = Path(ICON_ROOT / "select.png").__str__()
        self.toolbar._add_new_button(
            'select', "Enable or disable line selection", select_icon_file_path, "enable_selections", True)
        # Add span selection button to toolbar
        span_select_icon_file_path = Path(ICON_ROOT / "span_select.png").__str__()
        self.toolbar._add_new_button(
            'span_select', "Enable or disable span selection", span_select_icon_file_path, "enable_span_selections", True)
        # Insert the add_annotation
        add_annotation_icon_file_path = Path(ICON_ROOT / "add_annotation.png").__str__()
        self.toolbar._add_new_button(
            'add_annotation', "Add selected lines to current signal class", add_annotation_icon_file_path, "add_annotation", False)
        # Insert the delete_annotation
        delete_annotation_icon_file_path = Path(ICON_ROOT / "delete_annotation.png").__str__()
        self.toolbar._add_new_button(
            'delete_annotation', "Delete selected lines class annotation", delete_annotation_icon_file_path, "remove_annotation", False)

        # Create pick event connection id (used by line selector)
        self.pick_event_connection_id = None
        # Create mouse click event connection id (used to clear selections)
        self.mouse_click_event_connection_id = None
        # Set initial signal class valus to 0
        self._signal_class = 0
        # Initialize current_time_line
        self.vertical_time_line = None
        # Dictionary of valid overlays
        self.allowed_overlays = {
            'Annotations': {'display_markers': True},
            'Predictions': {'display_markers': False},
        }

        # Create horizontal Span Selector
        self.span_selector = SpanSelector(ax=self.axes,
                                          onselect=self._on_span_select,
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
        # Populate overlay combobox with allowed overlays
        self._populate_overlay_combobox()


        # z-step changed in viewer
        # Disconnect draw event on z-slider callback (improves performance)
        current_step_callbacks = self.viewer.dims.events.current_step.callbacks
        draw_callback_tuple = [callback for callback in current_step_callbacks if callback[1] == '_draw'][0]
        self.viewer.dims.events.current_step.disconnect(draw_callback_tuple)
        # Connect new slider callback
        self.viewer.dims.events.current_step.connect(self.on_dims_slider_change)
        
        # self.on_update_layers()
            
    def _populate_overlay_combobox(self) -> None:
        """Populate overlay combobox with valid overlays keys.
        """        
        # Clear overlay combobox
        while self._selectors['overlay'].count() > 0:
            self._selectors['overlay'].removeItem(0)
        # Add valid overlay keys for newly selected layer
        self._selectors['overlay'].addItems(self._get_valid_overlay_keys())
    
    def _get_valid_overlay_keys(self) -> Tuple[str, ...]:
        """Get valid overlays keys.

        Looks for table columns that are also present at allowed_overlay_keys dictionary.

        Returns
        -------
        Tuple[str, ...]
            Valid overlays keys.
        """
        if len(self.layers) == 0 or not (hasattr(self.layers[0], "features")):
            return []
        else:
            valid_keys = [key for key in self.layers[0].features.keys() if key in self.allowed_overlays.keys()]
            return valid_keys
        
    @property
    def overlay_axis_key(self) -> Union[str, None]:
        """
        Key for the overlay data.
        """
        if self._selectors["overlay"].count() == 0:
            return None
        else:
            return self._selectors["overlay"].currentText()

    @overlay_axis_key.setter
    def overlay_axis_key(self, key: str) -> None:
        self._selectors["overlay"].setCurrentText(key)
        self._draw()
        
    def on_overlay_checkbox_change(self, state) -> None:
        """Callback function for overlay checkbox change.

        Updates overlay combobox visibility.
        """
        self._selectors["overlay"].setVisible(self.show_overlay_checkbox.isChecked())
        self._draw()

    def on_dims_slider_change(self) -> None:
        pass
        # TO DO: update vertical line over plot (consider multithreading for performance, check details here:
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
                self.layers[0].events.show_selected_label.connect(self._show_selected_label)
                self.layers[0].events.selected_label.connect(self._show_selected_label)
            # Update overlay combobox
            if 'overlay' in self._selectors.keys():
                self._populate_overlay_combobox()

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
        self.colorBox.update()

    def enable_selections(self):
        """Enable or disable line selector.

        If enabled, span selector is disabled.
        """
        # Update toolbar buttons actions
        if self.toolbar._actions['select'].isChecked():
            self._enable_line_selector(True)
            # Disable span selector upon activation of line selector
            self.toolbar._actions['span_select'].setChecked(False)
            self._enable_span_selector(False)
        else:
            self._enable_line_selector(False)
        self.toolbar._update_buttons_checked()

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
            if self.pick_event_connection_id is not None:
                self.canvas.figure.canvas.mpl_disconnect(
                    self.pick_event_connection_id)
                self.pick_event_connection_id = None

    def enable_span_selections(self):
        """Enable or disable span selector.

        If enabled, line selector is disabled.
        """
        if self.toolbar._actions['span_select'].isChecked():
            self._enable_span_selector(True)
            # Disable line selector upon activation of span selector
            self.toolbar._actions['select'].setChecked(False)
            self._enable_line_selector(False)
        else:
            self._enable_span_selector(False)
        self.toolbar._update_buttons_checked()

    def _enable_span_selector(self, active=False):
        """
        Enable or disable span selector.

        If span selector was created, enable or disable it.
        """
        if self.span_selector is not None:
            self.toolbar._update_buttons_checked()
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

    def _on_span_select(self, xmin, xmax):
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
            if self.toolbar._actions['select'].isChecked():
                self._on_span_select(0, 0)
                self._clear_selections()
            # Right click resets span annotations if span selector is enabled
            elif self.toolbar._actions['span_select'].isChecked():
                self._on_span_select(0, 0)
            else:
                # resets plot colors (in case predictions colors were set)
                self.reset_plot_prediction_colors()

        elif event.button == 1:
            # If left-click with select tool enabled and shift key pressed, select all lines
            if self.toolbar._actions['select'].isChecked():
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
                span_mask = np.in1d(np.indices((len(line.annotations),)), line.span_indices)
                table_annotations[span_mask] = self._signal_class
            else:
                table_annotations[:] = self._signal_class
            # Update features and line annotations
            self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations'] = table_annotations
            line.annotations = table_annotations.values.tolist()
        # Show overlays
        self._selectors['overlay'].setCurrentText('Annotations')
        self.show_overlay_checkbox.setChecked(True)

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
                span_mask = np.in1d(np.indices((len(line.annotations),)), line.span_indices)
                table_annotations[span_mask] = 0
            else:
                table_annotations[:] = 0
            # Update features and line annotations
            self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations'] = table_annotations
            line.annotations = table_annotations.values.tolist()

    # def update_line_layout_from_column(self, column_name='Predictions', display_markers=False):
    #     """Update line layout (line collection) from a column in the features table.

    #     Line colors are used to display prediction values.

    #     Parameters
    #     ----------
    #     column_name : str
    #         Name of the column with results from a classification model.
    #     """
    #     for line in self._lines:
    #         label = line.label_from_napari_layer
    #         feature_table = self.layers[0].features
    #         # Get the annotation/predictions for the current object_id from table column
    #         list_of_values = feature_table[feature_table[self.object_id_axis_key] == label][column_name].values
    #         if display_markers:
    #             line.annotations = list_of_values
    #         else:
    #             line.predictions = list_of_values

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
        feature_table = feature_table.sort_values(by=[self.object_id_axis_key, self.x_axis_key])
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

            update_lines = True
            if len(self._lines) == 0:
                update_lines = False

            for j, (signal_x, signal_y) in enumerate(zip(x, y)):
                if self.layers[0].show_selected_label and j != self.layers[0].selected_label - 1:
                    continue
                label_name = self.y_axis_key

                if update_lines:
                    line = self._lines[j]
                    # Update line axes with current axes (in case axes were cleared)
                    line.axes = self.axes
                    line.set_data(signal_x, signal_y)
                else:
                    line = InteractiveLine2D(
                        xdata=signal_x, ydata=signal_y,
                        label_from_napari_layer=j + 1,
                        color_from_napari_layer=self.layers[0].get_color(j + 1),
                        color=self.layers[0].get_color(j + 1),
                        label=label_name,
                        linestyle='-',
                        picker=True,
                        pickradius=2,
                        alpha=0.7,
                        axes=self.axes,
                        canvas=self.figure.canvas)
                    self._lines += [line]
                # if show overlay is checked, draw annotations/predictions
                if hasattr(self, 'show_overlay_checkbox') and self.show_overlay_checkbox.isChecked():
                    if self.overlay_axis_key is not None:
                        label = line.label_from_napari_layer
                        feature_table = self.layers[0].features
                        # Get the annotation/predictions for the current object_id from table column
                        list_of_values = feature_table[feature_table[self.object_id_axis_key] == label][self.overlay_axis_key].values
                        display_markers = self.allowed_overlays[self.overlay_axis_key]['display_markers']
                        if display_markers:
                            line.annotations = list_of_values
                            line.annotations_visible = True
                        else:
                            line.predictions = list_of_values
                            line.predictions_visible = True
                else:
                    line.annotations_visible = False
                    line.predictions_visible = False

                # Add (or re-add) every line and scatter to axes (in case axes were cleared)
                line.add_to_axes()
            self.axes.set_xlabel(x_axis_name)
            self.axes.set_ylabel(y_axis_name)
            self.axes.autoscale(enable=True, axis='both', tight=True)
            self.canvas.draw()
