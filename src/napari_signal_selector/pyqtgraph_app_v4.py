import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from nap_plot_tools.tools import QtColorSpinBox
from nap_plot_tools import cat10_mod_cmap_first_transparent
from napari.layers import Labels
from matplotlib.colors import LinearSegmentedColormap
from psygnal import Signal

# Define ICON_ROOT (adjust this path as needed)
ICON_ROOT = Path(r"C:\Users\mazo260d\Documents\GitHub\napari-signal-selector\src\napari_signal_selector") / "icons"

# --- MyPlotWidget subclass for double-click selection and reduced grid opacity ---
class MyPlotWidget(pg.PlotWidget):
    def __init__(self, plotter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = plotter

    def mouseDoubleClickEvent(self, event):
        if any(curve.selected for curve in self.plotter.curves):
            for curve in self.plotter.curves:
                curve.selected = False
        else:
            for curve in self.plotter.curves:
                curve.selected = True
        event.accept()

# --- AnnotatedPlotDataItem definition with brush caching and new properties ---
class AnnotatedPlotDataItem(pg.PlotDataItem):
    # Class-level cache for brushes keyed by annotation index.
    _brush_cache = {}

    def __init__(self, *args, size=14, width=2, color='w', **kwargs):
        """
        A PlotDataItem subclass with two extra scatter overlays:
          - 'annotations': markers with a circle ("o")
          - 'predictions': markers with a "star" symbol.
        Clicking toggles its 'selected' property.
        Annotations and predictions are stored as indices into the cat10_mod_cmap_first_transparent colormap.
        New properties: width, color, and size.
        """
        super().__init__(*args, **kwargs)
        self._size = size
        self._selected = False
        self._width = width
        self._color = color

        # Default pen settings
        self.default_pen = pg.mkPen(self._color, width=self._width)
        self.highlight_pen = pg.mkPen(self._color, width=2*self._width)
        self.setPen(self.default_pen)

        self._default_marker_pen = pg.mkPen((150, 150, 150, 150), width=self._width)
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

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, w):
        self._width = w
        self.default_pen = pg.mkPen(self._color, width=self._width)
        self.highlight_pen = pg.mkPen(self._color, width=self._width * 2)
        self._highlight_marker_pen = pg.mkPen(self._color, width=self._width)
        self.setPen(self.default_pen)
        self.update_scatter_items()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, col):
        self._color = col
        self.default_pen = pg.mkPen(self._color, width=self._width)
        self.highlight_pen = pg.mkPen(self._color, width=self._width * 2)
        self._highlight_marker_pen = pg.mkPen(self._color, width=self._width)
        self.setPen(self.default_pen)
        self.update_scatter_items()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size):
        self._size = new_size
        self.annotation_scatter.setSize(new_size)
        self.prediction_scatter.setSize(new_size)
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
                    if val not in AnnotatedPlotDataItem._brush_cache:
                        color_tuple = cat10_mod_cmap_first_transparent(val)
                        qcolor = QtGui.QColor.fromRgbF(*color_tuple)
                        AnnotatedPlotDataItem._brush_cache[val] = pg.mkBrush(qcolor)
                    ann_brushes.append(AnnotatedPlotDataItem._brush_cache[val])
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
                    if val not in AnnotatedPlotDataItem._brush_cache:
                        color_tuple = cat10_mod_cmap_first_transparent(val)
                        qcolor = QtGui.QColor.fromRgbF(*color_tuple)
                        AnnotatedPlotDataItem._brush_cache[val] = pg.mkBrush(qcolor)
                    pred_brushes.append(AnnotatedPlotDataItem._brush_cache[val])
                    pred_pens.append(self._highlight_marker_pen if self.selected else self._default_marker_pen)
            self.prediction_scatter.setData(x=x, y=y, brush=pred_brushes, pen=pred_pens,
                                            symbol="star", size=self._size)

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, arr):
        self._annotations = np.array(arr, dtype=np.uint32)
        self.update_scatter_items()

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, arr):
        self._predictions = np.array(arr, dtype=np.uint32)
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

# --- PlotterWindow class with groupby caching, delta logging commit, and progress bar ---
class PlotterWindow(QtWidgets.QMainWindow):
    # Custom signal to indicate plot update finished.
    plot_update_finished = Signal(bool)
    def __init__(self, curve_colormap=None):
        super().__init__()
        self.setStatusBar(QtWidgets.QStatusBar())
        # Create a progress bar widget and add it to the status bar.
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self._updating_plot = False
        self.plot_update_finished.connect(self.updating_plot)

        self._df = None
        self._groups = None
        self._groups_key = None
        self._selectors = {}  # Dict[str, QComboBox] for keys "x", "y", "object_id"
        self._x_axis_key = None
        self._y_axis_key = None
        self._object_id_axis_key = None
        self.curves = []  # List to hold curves
        self.curve_colormap = curve_colormap  # External colormap (callable) for curve colors
        self.init_ui()
        self.init_signals()
        # Using timer to pdate dataframe every 10 seconds
        self.commit_timer = QtCore.QTimer(self)
        self.commit_timer.setInterval(10000)  # Update every 10 seconds
        self.commit_timer.timeout.connect(self.commit_annotation_changes)
        self.commit_timer.start()

    def init_ui(self):
        self.setWindowTitle("Plotter")
        self.resize(1000, 800)
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
        self.delete_all_action = QtWidgets.QAction(QtGui.QIcon(str(ICON_ROOT / "black" / "delete_all.png")),
                                                   "Delete All", self)
        self.toolbar.addAction(self.delete_all_action)
        self.toolbar.addSeparator()
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
        self.color_spin_box = QtColorSpinBox()
        control_layout.addWidget(QtWidgets.QLabel("Annotation Index:"))
        control_layout.addWidget(self.color_spin_box)
        control_layout.addStretch()
        central_layout.addWidget(control_panel)
        self.plot_widget = MyPlotWidget(self, title="Multiple Curves Demo")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
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
        self.delete_action.triggered.connect(lambda: self.delete_markers(annotations=True, predictions=False))
        self.delete_all_action.triggered.connect(lambda: self.delete_markers(annotations=True, predictions=True))
        self.show_selected_action.toggled.connect(self.toggle_show_selected)
        self.show_annotations_action.toggled.connect(self.toggle_annotations_visible)
        self.show_predictions_action.toggled.connect(self.toggle_predictions_visible)
        for key, combo in self._selectors.items():
            combo.activated.connect(lambda idx, k=key: self.on_selector_changed(k))

    def on_selector_changed(self, key):
        value = self._selectors[key].currentText()
        if key == "x":
            self.x_axis_key = value
        elif key == "y":
            self.y_axis_key = value
        elif key == "object_id":
            self.object_id_axis_key = value

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

    def updating_plot(self, value):
        self._updating_plot = not value

    def load_data_from_dataframe(self, df):
        self._df = df
        columns = list(df.columns)
        for key, combo in self._selectors.items():
            combo.clear()
            combo.addItems(columns)
        self.x_axis_key = "frame" if "frame" in columns else columns[0]
        self.object_id_axis_key = "label" if "label" in columns else (columns[1] if len(columns) > 1 else columns[0])
        self.statusBar().showMessage("Caching groupby result...", 3000)
        self._groups = self._df.groupby(self.object_id_axis_key)
        self._groups_key = self.object_id_axis_key
        for col in columns:
            if col not in (self.x_axis_key, self.object_id_axis_key):
                self.y_axis_key = col # setting y_axis_key will trigger update_plot_from_df if all other keys are already set
                break

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
        if self._groups is None or self._groups_key != obj_key:
            self.statusBar().showMessage("Re-grouping dataframe...", 3000)
            self._groups = df.groupby(obj_key)
            self._groups_key = obj_key
        groups_list = list(self._groups)
        self.progress_bar.setMaximum(len(groups_list))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        num_groups = len(groups_list)
        new_width = 1 if num_groups > 100 else 2
        self.plot_update_finished.emit(False)
        for i, (obj_id, group) in enumerate(groups_list):
            group_sorted = group.sort_values(by=x_key)
            x = group_sorted[x_key].values
            y = group_sorted[y_key].values
            item = AnnotatedPlotDataItem(x, y, antialias=False)
            item.width = new_width
            if self.curve_colormap is not None:
                norm_tuple = self.curve_colormap(obj_id)
                curve_color = QtGui.QColor.fromRgbF(*norm_tuple)
                item.default_pen = pg.mkPen(curve_color, width=new_width)
                item.setPen(item.default_pen)
            else:
                item.default_pen = pg.mkPen('w', width=new_width)
                item.setPen(item.default_pen)
            item.df_index = group_sorted.index
            if "annotation" in df.columns:
                item.annotations = group_sorted["annotation"].values
            else:
                item.annotations = np.zeros(len(x), dtype=np.uint32)
            if "prediction" in df.columns:
                item.predictions = group_sorted["prediction"].values
            else:
                item.predictions = np.zeros(len(x), dtype=np.uint32)
            self.plot_widget.addItem(item)
            self.curves.append(item)
            self.progress_bar.setValue(i + 1)
            QtWidgets.QApplication.processEvents()
        self.progress_bar.hide()
        self.statusBar().showMessage("Plot update complete.", 2000)
        self.plot_update_finished.emit(True)

    def toggle_region(self, enabled):
        if enabled:
            self.plot_widget.addItem(self.region_item)
        else:
            self.plot_widget.removeItem(self.region_item)
        self.region_enabled = enabled

    def apply_annotation(self):
        annotation_index = self.color_spin_box.get_color(norm=False)
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
                current = curve.annotations.copy() if curve.annotations is not None else np.full(len(x_data), 0, dtype=np.uint32)
                if self.region_enabled:
                    mask = (x_data >= region[0]) & (x_data <= region[1])
                    current[mask] = annotation_index
                else:
                    current[:] = annotation_index
                curve.annotations = current

    def delete_markers(self, annotations=True, predictions=True):
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
                    current_ann = curve.annotations.copy() if curve.annotations is not None else np.full(len(x_data), 0, dtype=np.uint32)
                    if self.region_enabled:
                        mask = (x_data >= region[0]) & (x_data <= region[1])
                        current_ann[mask] = 0
                    else:
                        current_ann[:] = 0
                    curve.annotations = current_ann
                if predictions:
                    current_pred = curve.predictions.copy() if curve.predictions is not None else np.full(len(x_data), 0, dtype=np.uint32)
                    if self.region_enabled:
                        mask = (x_data >= region[0]) & (x_data <= region[1])
                        current_pred[mask] = 0
                    else:
                        current_pred[:] = 0
                    curve.predictions = current_pred

    def delete_annotation(self):
        self.delete_markers(annotations=True, predictions=False)

    def delete_only_predictions(self):
        self.delete_markers(annotations=False, predictions=True)

    def delete_all_annotations(self):
        self.delete_markers(annotations=True, predictions=True)

    def commit_annotation_changes(self):
        if self._df is None:
            return
        # If plot is being updated, wait for it to finish before committing changes.
        if self._updating_plot:
            return
        self.statusBar().showMessage("Committing changes to dataframe...", 3000)
        total = len(self.curves)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        for i, curve in enumerate(self.curves):
            if hasattr(curve, 'df_index'):
                self._df.loc[curve.df_index, "annotation"] = curve.annotations
                self._df.loc[curve.df_index, "prediction"] = curve.predictions
            self.progress_bar.setValue(i+1)
            QtWidgets.QApplication.processEvents()
        self.progress_bar.hide()
        self.statusBar().showMessage("Dataframe updated.", 2000)

    def toggle_show_selected(self, checked):
        for curve in self.curves:
            curve.setVisible(curve.selected if checked else True)

    def toggle_annotations_visible(self, checked):
        for curve in self.curves:
            curve.annotations_visible = checked

    def toggle_predictions_visible(self, checked):
        for curve in self.curves:
            curve.predictions_visible = checked

    # @property
    # def x_axis_key(self):
    #     return self._x_axis_key

    # @x_axis_key.setter
    # def x_axis_key(self, value):
    #     self._x_axis_key = value
    #     self._selectors["x"].setCurrentText(value)
    #     if self._df is not None:
    #         self.update_plot_from_df()

    # @property
    # def y_axis_key(self):
    #     return self._y_axis_key

    # @y_axis_key.setter
    # def y_axis_key(self, value):
    #     self._y_axis_key = value
    #     self._selectors["y"].setCurrentText(value)
    #     if self._df is not None:
    #         self.update_plot_from_df()

    # @property
    # def object_id_axis_key(self):
    #     return self._object_id_axis_key

    # @object_id_axis_key.setter
    # def object_id_axis_key(self, value):
    #     self._object_id_axis_key = value
    #     self._selectors["object_id"].setCurrentText(value)
    #     if self._df is not None:
    #         self.update_plot_from_df()

if __name__ == '__main__':
  
    app = pg.mkQApp()
    from napari.utils import colormaps
    use_large_df = True

    # Obtain a colormap from a napari Labels layer.
    # labels_layer = Labels(np.zeros((2, 2), dtype=np.uint32))
    # colors = labels_layer.colormap.colors
    
    # win.init_signals()

    if use_large_df:

        # Set a random seed for reproducibility
        np.random.seed(42)

        # Parameters for the base signals
        n_periods = 3
        period = 20
        n_points = period * n_periods
        frames = np.arange(n_points)

        # Function to generate the base signal for a given signal type
        def base_signal(signal_type):
            if signal_type == 1:  # Sinusoidal wave
                return np.sin(2 * np.pi * (frames % period) / period)
            elif signal_type == 2:  # Square wave
                return np.sign(np.sin(2 * np.pi * (frames % period) / period))
            elif signal_type == 3:  # Saw-tooth wave
                return -1 + 2 * (frames % period) / (period - 1)
            elif signal_type == 4:  # Base noise pattern (to be modified later)
                noise_period = np.random.normal(0, 0.5, size=period)
                return np.tile(noise_period, n_periods)
            elif signal_type == 5:  # Quadratic polynomial
                t = np.linspace(0, period, period, endpoint=False)
                base_poly = 4 * ((t - period/2) / (period/2))**2 - 1
                return np.tile(base_poly, n_periods)
            else:
                return np.zeros(n_points)

        # Number of copies per signal type (total signals = 5 * copies)
        n_copies_per_signal = 200  # Adjust to generate more or fewer signals

        all_dfs = []
        label_counter = 1  # Counter for an increasing integer label

        for signal_type in range(1, 6):
            for copy in range(n_copies_per_signal):
                # Generate the base signal for the current type
                signal = base_signal(signal_type)
                # Generate a random offset and additional noise for this copy
                offset = np.random.uniform(-2, 2)
                noise_std = np.random.uniform(0, 0.5)
                additional_noise = np.random.normal(0, noise_std, size=signal.shape)
                # Create the modified signal by adding the offset and noise
                signal_modified = signal + offset + additional_noise
                
                # Use an increasing integer label
                current_label = label_counter
                label_counter += 1

                # Build a DataFrame for this signal copy with the integer label
                temp_df = pd.DataFrame({
                    "label": current_label,
                    "frame": frames,
                    "feature": signal_modified
                })
                all_dfs.append(temp_df)

        # Concatenate all individual DataFrames into one large DataFrame
        df = pd.concat(all_dfs, ignore_index=True)

        # Optionally add annotations and predictions with random values
        df["annotation"] = np.random.randint(0, 5, size=len(df), dtype=np.uint32)
        df["prediction"] = np.random.randint(0, 5, size=len(df), dtype=np.uint32)

    else:

        n_periods = 3
        period = 20
        n_points = period * n_periods
        frames = np.arange(n_points)

        y_sine = np.sin(2 * np.pi * (frames % period) / period)
        df1 = pd.DataFrame({"label": 1, "frame": frames, "feature": y_sine})

        y_square = np.sign(np.sin(2 * np.pi * (frames % period) / period))
        df2 = pd.DataFrame({"label": 2, "frame": frames, "feature": y_square})

        y_saw = -1 + 2 * (frames % period) / (period - 1)
        df3 = pd.DataFrame({"label": 3, "frame": frames, "feature": y_saw})

        np.random.seed(0)
        noise_period = np.random.normal(0, 0.5, size=period)
        y_noise = np.tile(noise_period, n_periods)
        df4 = pd.DataFrame({"label": 4, "frame": frames, "feature": y_noise})

        t = np.linspace(0, period, period, endpoint=False)
        y_poly_period = 4 * ((t - period/2) / (period/2))**2 - 1
        y_poly = np.tile(y_poly_period, n_periods)
        df5 = pd.DataFrame({"label": 5, "frame": frames, "feature": y_poly})

        df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
        df["annotation"] = np.random.randint(0, 5, size=len(df), dtype=np.uint32)
        df["prediction"] = np.random.randint(0, 5, size=len(df), dtype=np.uint32)

        label_counter = 5

    labels_layer_colormap = colormaps.label_colormap(
                label_counter + 1)
    colors = labels_layer_colormap.colors
    napari_labels_cmap = LinearSegmentedColormap.from_list("napari_labels_cmap", colors, N=len(colors))
    win = PlotterWindow(curve_colormap=napari_labels_cmap)

    win.load_data_from_dataframe(df)
    win.show()
    sys.exit(app.exec())
