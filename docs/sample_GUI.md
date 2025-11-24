# MM-Splat Dataset Viewer GUI Implementation

This document provides a detailed explanation of the MM-Splat dataset viewer GUI implementation.

## Overview

The GUI is built using the **PyQt6** framework. It provides a main window with several panes for visualizing the dataset, including a 3D world view, a range-doppler plot, and a control panel.

The core of the visualization is powered by the **pyqtgraph** library. **pyqtgraph.opengl** is used for the 3D visualization, and **pyqtgraph** is used for the 2D plotting.

The implementation follows a model-view-controller (MVC) pattern, where:

*   **Model**: The `ColmapDataset` class (`mm_splat/src/mm_splat/dataset_utils/datasets/colmap_dataset.py`) represents the data model. It is responsible for loading and providing access to the dataset.
*   **View**: The visualization components in the `mm_splat/src/mm_splat/visualization/visuals` directory and the main window (`mm_splat/src/mm_splat/visualization/gui/dataset_viewer_main_window.py`) represent the view. They are responsible for displaying the data.
*   **Controller**: The `DatasetViewerMainController` class (`mm_splat/src/mm_splat/visualization/backends/dataset_viewer_main_controller.py`) acts as the controller. It connects the model and the view, handling user input and updating the view when the data changes.

## Directory Structure

The relevant files and directories are:

```
mm_splat/
├── scripts/
│   └── launch_dataset_viewer.py   # Entry point
└── src/
    └── mm_splat/
        ├── dataset_utils/
        │   └── datasets/
        │       └── colmap_dataset.py  # Data model
        └── visualization/
            ├── backends/
            │   └── dataset_viewer_main_controller.py  # Controller
            ├── gui/
            │   ├── dataset_viewer_main_window.py    # Main window
            │   └── dataset_viewer_control_pannel.py # Control panel
            └── visuals/
                ├── world_view_3d.py                 # 3D view
                ├── range_doppler_view.py            # 2D plot
                ├── ...                              # Other view components
```

## Key Components

### 1. Entry Point

*   **File**: `mm_splat/scripts/launch_dataset_viewer.py`
*   **Purpose**: This script is the entry point for launching the GUI. It parses command-line arguments, initializes the `QApplication`, and creates the main window.

### 2. Main Window

*   **File**: `mm_splat/src/mm_splat/visualization/gui/dataset_viewer_main_window.py`
*   **Class**: `DatasetViewerMainWindow`
*   **Purpose**: This class defines the main application window. It creates the layout and assembles the different visualization components. It also creates an instance of the `DatasetViewerMainController`.

### 3. Controller

*   **File**: `mm_splat/src/mm_splat/visualization/backends/dataset_viewer_main_controller.py`
*   **Class**: `DatasetViewerMainController`
*   **Purpose**: This is the core of the application's logic. It:
    *   Loads the `ColmapDataset`.
    *   Creates instances of the visualization components (views).
    *   Connects the views to the data.
    *   Handles user interactions (e.g., changing the frame) and updates the views accordingly.
    *   Uses Qt's signal and slot mechanism to communicate with the views.

### 4. Data Model

*   **File**: `mm_splat/src/mm_splat/dataset_utils/datasets/colmap_dataset.py`
*   **Class**: `ColmapDataset`
*   **Purpose**: This class is responsible for loading and parsing the COLMAP dataset. It provides methods to access the point cloud, sensor poses, and other data for each frame.

### 5. Visualization Components (Views)

The `mm_splat/src/mm_splat/visualization/visuals` directory contains the individual components for visualizing the data. The two main components are:

*   **`WorldView3D` (`world_view_3d.py`)**:
    *   This view displays the 3D point cloud, sensor pose, and velocity.
    *   It uses `pyqtgraph.opengl.GLViewWidget` to create the 3D scene.
    *   It contains methods to update the pose, velocity, and scatter points.

*   **`RangeDopplerView` (`range_doppler_view.py`)**:
    *   This view displays the 2D range-doppler response as a heatmap.
    *   It uses `pyqtgraph.PlotWidget` and `pyqtgraph.ImageItem` to create the plot.
    *   It contains methods to set the axes and update the data.

Other views in this directory include `DatasetSelectionView`, `FrameUpdaterView`, and `MetadataView`.

## How to Replicate the GUI

To replicate this GUI in another repository, you would need to:

1.  **Dependencies**: Make sure you have the following key dependencies installed:
    *   `PyQt6`
    *   `pyqtgraph`
    *   `numpy`
    *   `scipy`

2.  **Copy/Adapt the Code**:
    *   Copy the `mm_splat/src/mm_splat/visualization` directory to your new project.
    *   Copy the `mm_splat/scripts/launch_dataset_viewer.py` script and adapt it to your project's structure.
    *   You will also need a data loader class similar to `ColmapDataset` that provides the data in the format expected by the controller and views.

3.  **Integrate with Your Data**:
    *   Modify the `DatasetViewerMainController` and the data loader to work with your dataset.
    *   Ensure that your data loader provides the necessary data, such as point clouds, sensor poses, and any other data you want to visualize.

By following this structure, you can create a similar GUI for your own datasets.
