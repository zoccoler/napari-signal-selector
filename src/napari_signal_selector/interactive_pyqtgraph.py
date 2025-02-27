import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from nap_plot_tools.tools import QtColorSpinBox
from nap_plot_tools import cat10_mod_cmap_first_transparent

# Define ICON_ROOT (adjust this path to your actual icons directory)
ICON_ROOT = Path(r"C:\Users\mazo260d\Documents\GitHub\napari-signal-selector\src\napari_signal_selector") / "icons"

# --- AnnotatedPlotDataItem definition (using indices for annotations) ---
class AnnotatedPlotDataItem(pg.PlotDataItem):
    def __init__(self, *args, size=14, width=2, color='w', **kwargs):
        """
        A PlotDataItem subclass with two extra scatter overlays:
          - 'annotations': markers with a circle ("o")
          - 'predictions': markers with a "star" symbol.
        Clicking on the curve or its markers toggles its 'selected' property.
        Annotations and predictions are stored as indices (integers) into the
        cat10_mod_cmap_first_transparent colormap.
        """
        super().__init__(*args, **kwargs)
        self._size = size
        self._width = width
        self._color = color
        self._selected = False

        self.default_pen = pg.mkPen(self._color, width=self._width)
        self.highlight_pen = pg.mkPen(self._color, width=2*self._width)
        self.setPen(self.default_pen)

        self._default_marker_pen = pg.mkPen(None)
        self._highlight_marker_pen = pg.mkPen(self._color, width=self._width)

        self.setCurveClickable(True, width=5)
        self.sigClicked.connect(self.on_clicked)

        self.annotation_scatter = pg.ScatterPlotItem()
        self.prediction_scatter = pg.ScatterPlotItem()
        self.annotation_scatter.setParentItem(self)
        self.prediction_scatter.setParentItem(self)
        self.annotation_scatter.sigClicked.connect(self.on_clicked)
        self.prediction_scatter.sigClicked.connect(self.on_clicked)

        self.annotation_scatter.setSymbol("o")
        self.prediction_scatter.setSymbol("star")

        self.annotation_scatter.setSize(self._size)
        self.prediction_scatter.setSize(self._size)

        self._annotations = None
        self._predictions = None

    def on_clicked(self, *args):
        self.selected = not self.selected

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        if value == self._size:
            return
        self._size = value
        self.annotation_scatter.setSize(self._size)
        self.prediction_scatter.setSize(self._size)

    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        if value == self._width:
            return
        self._width = value
        self.default_pen = pg.mkPen(self._color, width=self._width)
        self.highlight_pen = pg.mkPen(self._color, width=2*self._width)
        self.setPen(self.default_pen)

    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value):
        if value == self._color:
            return
        self._color = value
        self.default_pen = pg.mkPen(self._color, width=self._width)
        self.highlight_pen = pg.mkPen(self._color, width=2*self._width)
        self.setPen(self.default_pen)
        self._highlight_marker_pen = pg.mkPen(self._color, width=self._width)

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = value
        if value:
            self.setShadowPen(self.highlight_pen)
        else:
            self.setShadowPen(None)
        self.update_scatter_items()

    def update_scatter_items(self):
        data = self.getData()
        if data is None:
            return
        x, y = data
        n = len(x)
        transparent_brush = pg.mkBrush(QtGui.QColor(0, 0, 0, 0))
        transparent_pen = pg.mkPen(QtGui.QColor(0, 0, 0, 0))
        if self._annotations is not None:
            if len(self._annotations) != n:
                raise ValueError("Length of annotations array must equal number of data points.")
            ann_brushes = []
            ann_pens = []
            for val in self._annotations:
                if val == 0:
                    ann_brushes.append(transparent_brush)
                    ann_pens.append(transparent_pen)
                else:
                    # Get color from cat10_mod_cmap_first_transparent callable.
                    color_tuple = cat10_mod_cmap_first_transparent(val)
                    qcolor = QtGui.QColor.fromRgbF(*color_tuple)
                    ann_brushes.append(pg.mkBrush(qcolor))
                    ann_pens.append(self._highlight_marker_pen if self.selected else self._default_marker_pen)
            self.annotation_scatter.setData(x=x, y=y, brush=ann_brushes, pen=ann_pens,
                                              symbol="o", size=self._size)
        if self._predictions is not None:
            if len(self._predictions) != n:
                raise ValueError("Length of predictions array must equal number of data points.")
            pred_brushes = []
            pred_pens = []
            for val in self._predictions:
                if val == 0:
                    pred_brushes.append(transparent_brush)
                    pred_pens.append(transparent_pen)
                else:
                    color_tuple = cat10_mod_cmap_first_transparent(val)
                    qcolor = QtGui.QColor.fromRgbF(*color_tuple)
                    pred_brushes.append(pg.mkBrush(qcolor))
                    pred_pens.append(self._highlight_marker_pen if self.selected else self._default_marker_pen)
            self.prediction_scatter.setData(x=x, y=y, brush=pred_brushes, pen=pred_pens,
                                            symbol="star", size=self._size)

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, arr):
        self._annotations = np.array(arr, dtype=object)
        self.update_scatter_items()

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, arr):
        self._predictions = np.array(arr, dtype=object)
        self.update_scatter_items()

    @property
    def annotations_visible(self):
        return self.annotation_scatter.isVisible()

    @annotations_visible.setter
    def annotations_visible(self, value):
        self.annotation_scatter.setVisible(value)

    @property
    def predictions_visible(self):
        return self.prediction_scatter.isVisible()

    @predictions_visible.setter
    def predictions_visible(self, value):
        self.prediction_scatter.setVisible(value)

class MyPlotWidget(pg.PlotWidget):
    def __init__(self, plotter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = plotter

    def mouseDoubleClickEvent(self, event):
        # If any curves are selected, deselect all; otherwise, select all.
        if any(curve.selected for curve in self.plotter.curves):
            for curve in self.plotter.curves:
                curve.selected = False
        else:
            for curve in self.plotter.curves:
                curve.selected = True
        event.accept()

# --- PlotterWindow class ---
class PlotterWindow(QtWidgets.QMainWindow):
    def __init__(self, curve_colormap=None):
        super().__init__()
        self._df = None
        self._selectors = {}  # Dict[str, QComboBox] for "x", "y", "object_id"
        self._x_axis_key = None
        self._y_axis_key = None
        self._object_id_axis_key = None
        self.curves = []  # List to hold curves
        self.curve_colormap = curve_colormap  # External colormap for curve colors (optional)
        self.init_ui()
        self.init_signals()

    def init_ui(self):
        self.setWindowTitle("Plotter")
        self.resize(1000, 800)

        # Create toolbar.
        self.toolbar = self.addToolBar("main_toolbar")
        self.span_action = QtWidgets.QAction(self)
        span_icon = QtGui.QIcon()
        span_icon.addFile(str(ICON_ROOT / "black" / "span_select.png"),
                          mode=QtGui.QIcon.Normal, state=QtGui.QIcon.Off)
        span_icon.addFile(str(ICON_ROOT / "black" / "span_select_checked.png"),
                          mode=QtGui.QIcon.Normal, state=QtGui.QIcon.On)
        self.span_action.setIcon(span_icon)
        self.span_action.setText("Span Selector")
        self.span_action.setCheckable(True)
        self.toolbar.addAction(self.span_action)

        self.apply_action = QtWidgets.QAction(QtGui.QIcon(str(ICON_ROOT / "black" / "add_annotation.png")),
                                              "Apply Annotations", self)
        self.toolbar.addAction(self.apply_action)

        self.delete_action = QtWidgets.QAction(QtGui.QIcon(str(ICON_ROOT / "black" / "delete_annotation.png")),
                                               "Delete Annotations", self)
        self.toolbar.addAction(self.delete_action)

        self.show_selected_action = QtWidgets.QAction(self)
        show_selected_icon = QtGui.QIcon()
        show_selected_icon.addFile(str(ICON_ROOT / "black" / "show_all.png"),
                                   mode=QtGui.QIcon.Normal, state=QtGui.QIcon.Off)
        show_selected_icon.addFile(str(ICON_ROOT / "black" / "show_selected.png"),
                                   mode=QtGui.QIcon.Normal, state=QtGui.QIcon.On)
        self.show_selected_action.setIcon(show_selected_icon)
        self.show_selected_action.setText("Show Only Selected")
        self.show_selected_action.setCheckable(True)
        self.toolbar.addAction(self.show_selected_action)

        self.show_annotations_action = QtWidgets.QAction(self)
        show_anno_icon = QtGui.QIcon()
        show_anno_icon.addFile(str(ICON_ROOT / "black" / "hide_annotations.png"),
                               mode=QtGui.QIcon.Normal, state=QtGui.QIcon.Off)
        show_anno_icon.addFile(str(ICON_ROOT / "black" / "show_annotations.png"),
                               mode=QtGui.QIcon.Normal, state=QtGui.QIcon.On)
        self.show_annotations_action.setIcon(show_anno_icon)
        self.show_annotations_action.setText("Show Annotations")
        self.show_annotations_action.setCheckable(True)
        self.show_annotations_action.setChecked(True)
        self.toolbar.addAction(self.show_annotations_action)

        self.show_predictions_action = QtWidgets.QAction(self)
        show_pred_icon = QtGui.QIcon()
        show_pred_icon.addFile(str(ICON_ROOT / "black" / "hide_predictions.png"),
                               mode=QtGui.QIcon.Normal, state=QtGui.QIcon.Off)
        show_pred_icon.addFile(str(ICON_ROOT / "black" / "show_predictions.png"),
                               mode=QtGui.QIcon.Normal, state=QtGui.QIcon.On)
        self.show_predictions_action.setIcon(show_pred_icon)
        self.show_predictions_action.setText("Show Predictions")
        self.show_predictions_action.setCheckable(True)
        self.show_predictions_action.setChecked(True)
        self.toolbar.addAction(self.show_predictions_action)

        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.setContentsMargins(5, 5, 5, 5)
        self.setCentralWidget(central_widget)

        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        self.color_spin_box = QtColorSpinBox()  # Using QtColorSpinBox from nap_plot_tools
        control_layout.addWidget(QtWidgets.QLabel("Annotation Index:"))
        control_layout.addWidget(self.color_spin_box)
        control_layout.addStretch()
        central_layout.addWidget(control_panel)

        self.plot_widget = MyPlotWidget(self, title="Multiple Curves Demo")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.1)
        self.plot_widget.setLabel("left", "Y Value")
        self.plot_widget.setLabel("bottom", "X Value")
        central_layout.addWidget(self.plot_widget)

        self._selectors = {}
        selectors_panel = QtWidgets.QWidget()
        selectors_layout = QtWidgets.QHBoxLayout(selectors_panel)
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        for key in ["x", "y", "object_id"]:
            combo = QtWidgets.QComboBox()
            combo.setPlaceholderText(key)
            self._selectors[key] = combo
            selectors_layout.addWidget(QtWidgets.QLabel(key.upper() + ":"))
            selectors_layout.addWidget(combo)
        selectors_layout.addStretch()
        central_layout.addWidget(selectors_panel)

        self.region_item = pg.LinearRegionItem()
        self.region_item.setZValue(-10)
        self.region_enabled = False

        self._x_axis_key = None
        self._y_axis_key = None
        self._object_id_axis_key = None

    def init_signals(self):
        self.span_action.toggled.connect(self.toggle_region)
        self.apply_action.triggered.connect(self.apply_annotation)
        self.delete_action.triggered.connect(self.delete_markers)
        self.show_selected_action.toggled.connect(self.toggle_show_selected)
        self.show_annotations_action.toggled.connect(self.toggle_annotations_visible)
        self.show_predictions_action.toggled.connect(self.toggle_predictions_visible)
        for key, combo in self._selectors.items():
            combo.currentIndexChanged.connect(lambda idx, k=key: self.on_selector_changed(k))

    def on_selector_changed(self, key):
        value = self._selectors[key].currentText()
        if key == "x":
            self.x_axis_key = value
        elif key == "y":
            self.y_axis_key = value
        elif key == "object_id":
            self.object_id_axis_key = value
        if self._df is not None:
            self.update_plot_from_df()

    @property
    def x_axis_key(self):
        return self._x_axis_key

    @x_axis_key.setter
    def x_axis_key(self, value):
        self._x_axis_key = value
        self._selectors["x"].setCurrentText(value)
        if self._df is not None:
            self.update_plot_from_df()

    @property
    def y_axis_key(self):
        return self._y_axis_key

    @y_axis_key.setter
    def y_axis_key(self, value):
        self._y_axis_key = value
        self._selectors["y"].setCurrentText(value)
        if self._df is not None:
            self.update_plot_from_df()

    @property
    def object_id_axis_key(self):
        return self._object_id_axis_key

    @object_id_axis_key.setter
    def object_id_axis_key(self, value):
        self._object_id_axis_key = value
        self._selectors["object_id"].setCurrentText(value)
        if self._df is not None:
            self.update_plot_from_df()

    def load_data_from_dataframe(self, df):
        """Load a pandas DataFrame, populate combo boxes, and plot curves.
           Expected columns: object_id, x (e.g. frame), y (e.g. feature),
           and optionally 'annotation' and 'prediction'.
        """
        self._df = df
        columns = list(df.columns)
        for key, combo in self._selectors.items():
            combo.clear()
            combo.addItems(columns)
        self.x_axis_key = "frame" if "frame" in columns else columns[0]
        self.object_id_axis_key = "label" if "label" in columns else (columns[1] if len(columns) > 1 else columns[0])
        for col in columns:
            if col not in (self.x_axis_key, self.object_id_axis_key):
                self.y_axis_key = col
                break
        self.update_plot_from_df()

    def update_plot_from_df(self):
        for item in self.curves:
            self.plot_widget.removeItem(item)
        self.curves = []
        df = self._df
        if df is None:
            return
        x_key = self.x_axis_key
        y_key = self.y_axis_key
        obj_key = self.object_id_axis_key
        if not (x_key and y_key and obj_key):
            return
        self.plot_widget.setLabel("bottom", x_key)
        self.plot_widget.setLabel("left", y_key)
        groups = df.groupby(obj_key)
        for obj_id, group in groups:
            group_sorted = group.sort_values(by=x_key)
            x = group_sorted[x_key].values
            y = group_sorted[y_key].values
            item = AnnotatedPlotDataItem(x, y, antialias=False)
            # Use the external curve_colormap if provided.
            if self.curve_colormap is not None:
                # Assume the curve_colormap.colormap callable returns a normalized rgba tuple.
                norm_tuple = self.curve_colormap(obj_id)
                curve_color = QtGui.QColor.fromRgbF(*norm_tuple)
                item.color = curve_color
                # item.default_pen = pg.mkPen(curve_color, width=item.width)
                # item.setPen(item.default_pen)
            else:
                item.color = 'w'
                # item.default_pen = pg.mkPen('w', width=item.width)
                # item.setPen(item.default_pen)
            if len(groups) > 10E3:
                item.width = 1
            else:
                item.width = 2
            if "annotation" in df.columns:
                item.annotations = group_sorted["annotation"].values
            else:
                item.annotations = np.zeros(len(x), dtype=object)
            if "prediction" in df.columns:
                item.predictions = group_sorted["prediction"].values
            else:
                item.predictions = np.zeros(len(x), dtype=object)
            self.plot_widget.addItem(item)
            self.curves.append(item)

    def toggle_region(self, enabled):
        if enabled:
            self.plot_widget.addItem(self.region_item)
        else:
            self.plot_widget.removeItem(self.region_item)
        self.region_enabled = enabled

    def apply_annotation(self):
        # Use get_color(norm=False) to get an integer index.
        annotation_index = self.color_spin_box.value  # Returns an integer index.
        if not self.curves:
            return
        if self.region_enabled:
            region = self.region_item.getRegion()
        for curve in self.curves:
            if curve.selected:
                data = curve.getData()
                if data is None:
                    continue
                x_data = data[0]
                current = curve.annotations.copy() if curve.annotations is not None else np.full(len(x_data), 0, dtype=object)
                if self.region_enabled:
                    mask = (x_data >= region[0]) & (x_data <= region[1])
                    current[mask] = annotation_index
                else:
                    current[:] = annotation_index
                curve.annotations = current

    def delete_markers(self, checked, annotations=True, predictions=True):
        if not self.curves:
            return
        if self.region_enabled:
            region = self.region_item.getRegion()
        for curve in self.curves:
            if curve.selected:
                data = curve.getData()
                if data is None:
                    continue
                x_data = data[0]
                if annotations:
                    current = curve.annotations.copy() if curve.annotations is not None else np.full(len(x_data), 0, dtype=object)
                    if self.region_enabled:
                        mask = (x_data >= region[0]) & (x_data <= region[1])
                        current[mask] = 0
                    else:
                        current[:] = 0
                    curve.annotations = current
                if predictions:
                    current = curve.predictions.copy() if curve.predictions is not None else np.full(len(x_data), 0, dtype=object)
                    if self.region_enabled:
                        mask = (x_data >= region[0]) & (x_data <= region[1])
                        current[mask] = 0
                    else:
                        current[:] = 0
                    curve.predictions = current

    def toggle_show_selected(self, checked):
        for curve in self.curves:
            curve.setVisible(curve.selected if checked else True)

    def toggle_annotations_visible(self, checked):
        for curve in self.curves:
            curve.annotations_visible = checked

    def toggle_predictions_visible(self, checked):
        for curve in self.curves:
            curve.predictions_visible = checked

    @property
    def x_axis_key(self):
        return self._x_axis_key

    @x_axis_key.setter
    def x_axis_key(self, value):
        self._x_axis_key = value
        self._selectors["x"].setCurrentText(value)
        if self._df is not None:
            self.update_plot_from_df()

    @property
    def y_axis_key(self):
        return self._y_axis_key

    @y_axis_key.setter
    def y_axis_key(self, value):
        self._y_axis_key = value
        self._selectors["y"].setCurrentText(value)
        if self._df is not None:
            self.update_plot_from_df()

    @property
    def object_id_axis_key(self):
        return self._object_id_axis_key

    @object_id_axis_key.setter
    def object_id_axis_key(self, value):
        self._object_id_axis_key = value
        self._selectors["object_id"].setCurrentText(value)
        if self._df is not None:
            self.update_plot_from_df()

if __name__ == '__main__':
    app = pg.mkQApp()

    from napari.layers import Labels
    from matplotlib.colors import LinearSegmentedColormap
    import napari

    viewer = napari.Viewer()

    labels_layer = Labels(np.zeros((2, 2), dtype=np.uint32))

    viewer.add_layer(labels_layer)

    colors = labels_layer.colormap.colors
    napari_labels_cmap = LinearSegmentedColormap.from_list("napari_labels_cmap", colors, N=len(colors))

    widget = PlotterWindow(curve_colormap=napari_labels_cmap)
    widget.init_signals()

    # Create a sample DataFrame with five types of signals.
    n_periods = 3
    period = 20
    n_points = period * n_periods
    frames = np.arange(n_points)

    # Signal 1: Sinusoidal wave.
    y_sine = np.sin(2 * np.pi * (frames % period) / period)
    df1 = pd.DataFrame({"label": 1, "frame": frames, "feature": y_sine})

    # Signal 2: Square wave.
    y_square = np.sign(np.sin(2 * np.pi * (frames % period) / period))
    df2 = pd.DataFrame({"label": 2, "frame": frames, "feature": y_square})

    # Signal 3: Saw-tooth wave.
    y_saw = -1 + 2 * (frames % period) / (period - 1)
    df3 = pd.DataFrame({"label": 3, "frame": frames, "feature": y_saw})

    # Signal 4: Random noise.
    np.random.seed(0)
    noise_period = np.random.normal(0, 0.5, size=period)
    y_noise = np.tile(noise_period, n_periods)
    df4 = pd.DataFrame({"label": 4, "frame": frames, "feature": y_noise})

    # Signal 5: Quadratic polynomial.
    t = np.linspace(0, period, period, endpoint=False)
    y_poly_period = 4 * ((t - period/2) / (period/2))**2 - 1
    y_poly = np.tile(y_poly_period, n_periods)
    df5 = pd.DataFrame({"label": 5, "frame": frames, "feature": y_poly})

    # Concatenate signals.
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    # Add new columns for annotations and predictions as indices (0 means no annotation).
    df["annotation"] = np.random.randint(0, 5, size=len(df))
    df["prediction"] = np.random.randint(0, 5, size=len(df))

    widget.load_data_from_dataframe(df)

    viewer.window.add_dock_widget(widget, area="right")
    napari.run()

    # win.show()
    # sys.exit(app.exec())
