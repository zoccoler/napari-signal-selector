from typing import Optional, Tuple, Any
import napari
import numpy as np
import numpy.typing as npt
from pathlib import Path
from matplotlib.lines import Line2D
# from matplotlib.pyplot import scatter
from matplotlib.colors import ListedColormap, Normalize
from napari_matplotlib.line import FeaturesLineWidget
from napari_matplotlib.base import NapariNavigationToolbar
from napari_matplotlib.util import Interval
from napari_signal_selector.utilities import get_custom_cat10based_cmap_list
from matplotlib.widgets import SpanSelector
from qtpy.QtWidgets import QWidget, QSpinBox, QLabel, QHBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QColor, QPainter

from matplotlib.backends.qt_compat import  _enum, _to_int
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
    _default_marker_size = 4

    def __init__(self, *args, axes=None, canvas=None, label_from_napari_layer, color_from_napari_layer,
                 selected=False, annotations=None, categorical_color=None, span_indices=[], **kwargs, ):
        super().__init__(*args, **kwargs)
        self._axes = axes
        self._canvas = canvas
        self.label_from_napari_layer = label_from_napari_layer
        self.color_from_napari_layer = color_from_napari_layer
        self._selected = selected
        self._annotations = annotations
        if self._annotations is None:
            self._annotations = np.ones(self.get_xdata().shape).tolist()
        self._categorical_color = categorical_color
        self._span_indices = span_indices
        if self._axes:
            self._annotations_scatter = self._axes.scatter(
                self.get_xdata(), self.get_ydata(), color='red')#c=self._annotations, cmap=self.mpl_cmap, norm=self.normalizer,)
                # marker='o', s=self._default_marker_size) 
            # self._axes.scatter([1,2,3,4,5], [0,1,1,2,2], color='red')
            self.scatter = self._axes.scatter([0,200], [0,10], color='red', s=400)
        else:
            self._annotations_scatter = None
        a=1


    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = value
        if value == True:
            self.set_linestyle('--')
            self.set_alpha(1)
        elif value == False:
            self.set_linestyle('-')
            self.set_alpha(self._default_alpha)
        self._canvas.draw_idle()

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, list_of_values):
        # TO DO
        # Feed value to indices given by span_indices
        # if span_indices is empty, feed entire annotations with value
        # update annotations_scatter 'c' parameter with self._annotations
        # draw idle
        # if self._span_indices.any():
        #     span_mask = np.in1d(np.indices(self._annotations.shape), self._span_indices)
        #     self._annotations[span_mask] = value
        # else:
        #     self._annotations[:] = value
        self._annotations = list_of_values
        # Update scatter plot x and y coordinates
        self._annotations_scatter.set_offsets(self.get_xydata().tolist())
        # Update scatter plot array with annotations (which yield marker colors)
        # self._annotations_scatter.set_array(self._annotations)
        self._canvas.draw_idle()
        # Update colors of scatter plot (markers)
        # Find a way to update 'c'
        # self._annotations_scatter.set?

        # self._annotation = value
        # if value > 0:
        #     self.set_marker('o')
        #     self.set_markersize(self._default_marker_size)
        #     annotation_color = self.cmap[value]
        #     self.set_markeredgecolor(annotation_color)
        #     self.set_markeredgewidth(1)
        # elif value == 0:
        #     self.set_marker('None')
        # self.figure.canvas.draw_idle()

    @property
    def categorical_color(self):
        return self._categorical_color

    @categorical_color.setter
    def categorical_color(self, value):
        self._categorical_color = value
        if value is not None:
            color = self.cmap[value]
            self.set_color(color)
        else:
            # Restore original line color
            self.set_color(self.color_from_napari_layer)
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
                xydata = self.get_xydata()
                self.scatter.set_offsets(xydata.tolist())
                self.scatter.set_color('green')
                self.scatter.set_sizes(np.repeat([400], len(xydata)))
                # self._axes.autoscale(enable=True, axis='both')
    def add_to_axes(self):
        if self._axes:
            self._axes.add_line(self)
            self._axes.add_artist(self.scatter)

    # def set_xdata(self, x):
    #     super().set_xdata(x)
    #     if hasattr(self, '_annotations_scatter'):
    #         y = self.get_ydata()
    #         self._annotations_scatter.set_offsets(list(zip(x, y)))
    #         self._canvas.draw_idle()

    # def set_ydata(self, y):
    #     super().set_ydata(y)
    #     if hasattr(self, '_annotations_scatter'):
    #         x = self.get_xdata()
    #         self._annotations_scatter.set_offsets(list(zip(x, y)))
    #         self._canvas.draw_idle()

    @property
    def axes(self):
        print("Accessing axes")
        print(self._axes)
        return self._axes
    
    @axes.setter
    def axes(self, value):
        print("Modifying axes")
        self._axes = value
        print(self._axes)


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

    def _add_new_button(self, text, tooltip_text, icon_image_file_path, callback, checkable=False, separator=True):
        """Add a new buttons to the toolbar.

        Parameters
        ----------
        text : str
            the text representing the name of the button
        tooltip_text : str
            the tooltip text exhibited when cursor hovers over button
        icon_image_file_path : str
            path to the "png" file containing the button image
        callback : function
            function to be called when button is clicked
        separator: bool
            Whether to add a separator before new button
        checkable: bool
            flag that indicates if button should or not be chackable
        """        
        self.extra_button_paths.append(icon_image_file_path)
        self.toolitems.append((text, tooltip_text, icon_image_file_path, callback))
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
        a = self.addAction(QIcon(icon_image_file_path), text, callback)
        self._actions[text] = a
        if checkable:
            a.setCheckable(True)
        if tooltip_text is not None:
            a.setToolTip(tooltip_text)
       
        ## Rebuild spacer at the very end of toolbar (re-create 'locLabel' created by __init__ from NavigationToolbar2QT)
        # https://github.com/matplotlib/matplotlib/blob/85d7bb370186f2fa86df6ecc3d5cd064eb7f0b45/lib/matplotlib/backends/backend_qt.py#L631
        if self.tb_coordinates:
            self.locLabel = QLabel("", self)
            self.locLabel.setAlignment(Qt.AlignmentFlag(
                _to_int(_enum("QtCore.Qt.AlignmentFlag").AlignRight) |
                _to_int(_enum("QtCore.Qt.AlignmentFlag").AlignVCenter)))
            self.locLabel.setSizePolicy(QSizePolicy(
                _enum("QtWidgets.QSizePolicy.Policy").Expanding,
                _enum("QtWidgets.QSizePolicy.Policy").Ignored,
            ))
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

        signal_selection_box = QHBoxLayout()
        signal_selection_box.addWidget(label)
        signal_selection_box.addLayout(self.signal_class_color_spinbox)
        self.layout().insertLayout(2, signal_selection_box)

        # Create an instance of your custom toolbar
        custom_toolbar = CustomNapariNavigationToolbar(self.canvas, parent=self)
        # Replace the default toolbar with the custom one
        self.setCustomToolbar(custom_toolbar)
        self._replace_toolbar_icons()

        # Add span selection button to toolbar
        select_icon_file_path = Path(ICON_ROOT / "select.png").__str__()
        self.toolbar._add_new_button(
            'select', "Enable or disable line selection", select_icon_file_path, self.enable_selections, True)
        # Add span selection button to toolbar
        span_select_icon_file_path = Path(ICON_ROOT / "span_select.png").__str__()
        self.toolbar._add_new_button(
            'span_select', "Enable or disable span selection", span_select_icon_file_path, self.enable_span_selections, True)
        # Insert the add_annotation
        add_annotation_icon_file_path = Path(ICON_ROOT / "add_annotation.png").__str__()
        self.toolbar._add_new_button(
            'add_annotation', "Add selected lines to current signal class", add_annotation_icon_file_path, self.add_annotation, False)
        # Insert the delete_annotation
        delete_annotation_icon_file_path = Path(ICON_ROOT / "delete_annotation.png").__str__()
        self.toolbar._add_new_button(
            'delete_annotation', "Delete selected lines class annotation", delete_annotation_icon_file_path, self.remove_annotation, False)

        # Create pick event connection id (used by line selector)
        self.pick_event_connection_id = None
        # Create mouse click event connection id (used to clear selections)
        self.mouse_click_event_connection_id = None
        # Set initial signal class valus to 0
        self._signal_class = 0

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

    def on_update_layers(self) -> None:
        """
        Called when the layer selection changes by ``self.update_layers()``.
        """
        super().on_update_layers()
        if len(self.layers) > 0:
            if 'show_selected_label' in self.layers[0].events.emitters.keys():
                self.layers[0].events.show_selected_label.connect(self._show_selected_label)
                self.layers[0].events.selected_label.connect(self._show_selected_label)

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
            if self.toolbar._actions['span_select'].isChecked():
                self._on_span_select(0, 0)
            # resets plot colors (in case categoriy color were set)
            self.reset_plot_colors()

        elif event.button == 1:
            # If left-click with select tool enabled and shift key pressed, select all lines
            if self.toolbar._actions['select'].isChecked():
                if modifiers == Qt.ShiftModifier:
                    self._select_all_lines()

            self.reset_plot_colors()

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
            self.layers[0].features.loc[
                self.layers[0].features[self.object_id_axis_key] == line.label_from_napari_layer, 'Annotations'] = 0
            line.annotations = 0

    def update_line_layout_from_column(self, column_name='Annotations', markers=False):
        """Update line layout (color or annotation) from a column in the features table.

        Markers are used to display annotations, while line colors are used to display categorical values.

        Parameters
        ----------
        column_name : str
            Name of the column with annotations or with results from a classification model.
        markers : bool, optional
            If True, update line annotations, if False, update line categorical colors, by default False.
        """
        for line in self._lines:
            label = line.label_from_napari_layer
            feature_table = self.layers[0].features
            # table = self.viewer.layers.selection.active.features
            # Get the annotation for the current object_id from table column
            values = feature_table[feature_table[self.object_id_axis_key] == label][column_name].values
            # If all values are the same, use that value
            if np.all(values == values[0]):
                value = values[0]
            else:
                # If not, get the most frequent non-zero annotation
                unique_values, counts = np.unique(values, return_counts=True)
                # Get the most frequent annotation
                value = values[np.argmax(counts[1:])]  # exclude 0
                indices = np.where(values == value)[0]
            if markers:
                line.annotations = value
                line.span_indices = indices
            else:
                line.categorical_color = value

    def reset_plot_colors(self):
        """Reset plot colors to original colors from napari layer (remove categorical colors).
        """
        for line in self._lines:
            line.categorical_color = None
        return

    def reset_plot_annotations(self):
        """Reset plot annotations to 0 (remove annotations).
        """
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

        # Sort features by object_id and x_axis_key
        feature_table = feature_table.sort_values(by=[self.object_id_axis_key, self.x_axis_key])
        # Get data for each object_id (usually label)
        grouped = feature_table.groupby(self.object_id_axis_key)
        x = np.array([sub_df[self.x_axis_key].values for label, sub_df in grouped]).T.squeeze()
        y = np.array([sub_df[self.y_axis_key].values for label, sub_df in grouped]).T.squeeze()

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

            for j, (signal_x, signal_y) in enumerate(zip(x.T, y.T)):
                if self.layers[0].show_selected_label and j != self.layers[0].selected_label - 1:
                    continue
                label_name = self.y_axis_key

                if update_lines:
                    line = self._lines[j]
                    line.axes = self.axes
                    # line.set_xdata(signal_x)
                    # line.set_ydata(signal_y)
                    line.set_data(signal_x, signal_y)
                    # self.axes.add_line(line)
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
                    # line.set_data(signal_x, signal_y)                    
                    # self.axes.add_line(line)
                    
                    self._lines += [line]
                line.add_to_axes()
            self.axes.set_xlabel(x_axis_name)
            self.axes.set_ylabel(y_axis_name)
            self.axes.autoscale(enable=True, axis='both', tight=True)
            self.apply_napari_colorscheme(self.axes)
            self.canvas.draw()
