import sys
import numpy as np
from vispy import scene, app
from vispy.scene.widgets import AxisWidget
from nap_plot_tools import cat10_mod_cmap_first_transparent

class CompositeSignalVisual(scene.Node):
    """
    CompositeSignalVisual is a custom visual that encapsulates a line and two marker visuals.
    
    It displays a signal (given as x and y numpy arrays) and supports interactivity:
      - A Line visual represents the signal.
      - Two Markers visuals are drawn:
          - predictions_marker: drawn with a star symbol.
          - annotations_marker: drawn with a filled circle symbol.
      
    Properties:
      - annotations: an array of positive integers (clipped at 5) used to determine marker colors
                     via the provided colormap.
      - predictions: similar to annotations for the predictions_marker.
      - size: marker size (default: 24).
      - width: base line width (default: 3); when selected, the width is doubled.
      - color: the line color (default: 'white').
      - selected: a boolean flag; when True, the line width is doubled.
    
    Mouse clicks on any child visual (line or markers) toggle the selected state.
    """
    def __init__(self, id, x, y, colormap, parent=None, size=24, width=3, color='white'):
        super().__init__(parent=parent)
        self.id = id
        self.data = np.column_stack((x, y))
        self.colormap = colormap

        # Initialize properties.
        self._size = size         # Marker size.
        self._width = width       # Base line width.
        self._color = color       # Line color.
        self._selected = False    # Selection state.

        # Initialize annotations and predictions (default zeros).
        self._annotations = np.zeros(self.data.shape[0], dtype=int)
        self._predictions = np.zeros(self.data.shape[0], dtype=int)

        # Create the line visual.
        self.line = scene.visuals.Line(
            self.data,
            color=self._color,
            width=self._width,
            parent=self
        )
        self.line.interactive = True
        self.line.events.mouse_press.connect(self.on_child_mouse_press)

        # Create the predictions markers visual.
        self.predictions_marker = scene.visuals.Markers(parent=self)
        self.predictions_marker.interactive = True
        self.predictions_marker.events.mouse_press.connect(self.on_child_mouse_press)

        # Create the annotations markers visual.
        self.annotations_marker = scene.visuals.Markers(parent=self)
        self.annotations_marker.interactive = True
        self.annotations_marker.events.mouse_press.connect(self.on_child_mouse_press)

        # Update marker visuals.
        self._update_predictions_marker()
        self._update_annotations_marker()

    def _update_annotations_marker(self):
        """Update the annotations marker based on the current annotations data."""
        annotations = np.clip(self._annotations, 0, 5)
        colors = np.array([self.colormap(i) for i in annotations])
        self.annotations_marker.set_data(self.data, face_color=colors, size=self._size, symbol='o')

    def _update_predictions_marker(self):
        """Update the predictions marker based on the current predictions data."""
        predictions = np.clip(self._predictions, 0, 5)
        colors = np.array([self.colormap(i) for i in predictions])
        self.predictions_marker.set_data(self.data, face_color=colors, size=self._size, symbol='star')

    @property
    def annotations(self):
        """Get or set the annotations array."""
        return self._annotations

    @annotations.setter
    def annotations(self, value):
        if len(value) != self.data.shape[0]:
            raise ValueError("Length of annotations must match number of data points.")
        self._annotations = np.clip(np.array(value), 0, 5)
        self._update_annotations_marker()

    @property
    def predictions(self):
        """Get or set the predictions array."""
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        if len(value) != self.data.shape[0]:
            raise ValueError("Length of predictions must match number of data points.")
        self._predictions = np.clip(np.array(value), 0, 5)
        self._update_predictions_marker()

    @property
    def selected(self):
        """Get or set the selected state. When selected, the line width is doubled."""
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = bool(value)
        new_width = self._width * 2 if self._selected else self._width
        self.line.set_data(self.data, color=self._color, width=new_width)
        print(f"Signal {self.id} selected: {self._selected}")

    @property
    def size(self):
        """Get or set the marker size."""
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self._update_predictions_marker()
        self._update_annotations_marker()

    @property
    def width(self):
        """Get or set the base line width."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        new_width = self._width * 2 if self._selected else self._width
        self.line.set_data(self.data, color=self._color, width=new_width)

    @property
    def color(self):
        """Get or set the line color."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        new_width = self._width * 2 if self._selected else self._width
        self.line.set_data(self.data, color=self._color, width=new_width)

    def on_child_mouse_press(self, event):
        """
        Callback for mouse press events on the child visuals.
        Toggles the selected state.
        """
        print(f"Child visual received mouse press at: {event.pos}")
        self.selected = not self.selected
        event.handled = True


# --------------------- Layout and Example Usage ---------------------
if __name__ == '__main__':
    # Create a SceneCanvas with a fixed size.
    canvas = scene.SceneCanvas(keys='interactive', size=(600, 600), show=True, bgcolor='black')
    
    # Create a grid layout with a margin.
    grid = canvas.central_widget.add_grid(margin=10)
    grid.spacing = 0

    # Add a title label spanning two columns.
    title = scene.Label("Plot Title", color='white')
    title.height_max = 40
    grid.add_widget(title, row=0, col=0, col_span=2)

    # Add the Y-axis widget on the left with the specified settings.
    yaxis = scene.AxisWidget(
        orientation='left',
        axis_label='Y Axis',
        axis_font_size=12,
        axis_label_margin=70,
        tick_label_margin=15
    )
    yaxis.width_max = 120
    grid.add_widget(yaxis, row=1, col=0)

    # Add the X-axis widget at the bottom with the specified settings.
    xaxis = scene.AxisWidget(
        orientation='bottom',
        axis_label='X Axis',
        axis_font_size=12,
        axis_label_margin=80,
        tick_label_margin=45
    )
    xaxis.height_max = 120
    grid.add_widget(xaxis, row=2, col=1)

    # Add right padding.
    right_padding = grid.add_widget(row=1, col=2, row_span=1)
    right_padding.width_max = 50

    # Create the main view and add it to the grid.
    view = grid.add_view(row=1, col=1, border_color='white')
    view.camera = 'panzoom'

    # Create multiple CompositeSignalVisual instances.
    signals = []
    n_signals = 5
    for i in range(n_signals):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i) + i * 2
        signal_vis = CompositeSignalVisual(
            id=i,
            x=x,
            y=y,
            colormap=cat10_mod_cmap_first_transparent,
            parent=view.scene,
            size=24,
            width=3,
            color='white'
        )
        # Assign random annotations and predictions (values between 0 and 5).
        annotations = np.random.randint(0, 6, size=len(x))
        predictions = np.random.randint(0, 6, size=len(x))
        signal_vis.annotations = annotations
        signal_vis.predictions = predictions

        if i == 0:
            signal_vis.selected = True

        signals.append(signal_vis)

    # Compute the overall bounding box without concatenating data.
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    for s in signals:
        local_min = s.data.min(axis=0)
        local_max = s.data.max(axis=0)
        if local_min[0] < min_x:
            min_x = local_min[0]
        if local_min[1] < min_y:
            min_y = local_min[1]
        if local_max[0] > max_x:
            max_x = local_max[0]
        if local_max[1] > max_y:
            max_y = local_max[1]

    # Compute margins (10% of each range).
    margin_x = (max_x - min_x) * 0.1
    margin_y = (max_y - min_y) * 0.1

    # Set the camera range explicitly to include all signals.
    view.camera.set_range(
        x=(min_x - margin_x, max_x + margin_x),
        y=(min_y - margin_y, max_y + margin_y)
    )

    # Link axes to the main view.
    xaxis.link_view(view)
    yaxis.link_view(view)

    if sys.flags.interactive == 0:
        app.run()
