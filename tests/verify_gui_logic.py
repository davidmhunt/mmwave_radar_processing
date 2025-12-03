import sys
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QComboBox

# Import views
from mmwave_radar_processing.visualization.views.range_angle_view import RangeAngleView
from mmwave_radar_processing.visualization.views.micro_doppler_view import MicroDopplerView
from mmwave_radar_processing.visualization.views.doppler_azimuth_view import DopplerAzimuthView
from mmwave_radar_processing.visualization.views.range_doppler_view import RangeDopplerView
from mmwave_radar_processing.visualization.views.range_response_view import RangeResponseView

# Fixture for QApplication
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_range_angle_view(qapp):
    view = RangeAngleView()
    # Data: [range, angle] = [10, 20]
    data = np.random.rand(10, 20)
    payload = {
        "data": data,
        "range_bins": np.linspace(0, 10, 10),
        "angle_bins": np.linspace(-1, 1, 20)
    }
    view.set_data(payload)
    
    # Check image shape: should be [angle, range] = [20, 10] due to transpose
    image_data = view.image.image
    assert image_data.shape == (20, 10)
    assert np.allclose(image_data, data.T)
    
    # Test dB mode
    view.set_db_mode(True)
    image_data_db = view.image.image
    expected_db = 20 * np.log10(np.maximum(np.abs(data.T), 1e-12))
    assert np.allclose(image_data_db, expected_db)
    
    # Test colormap
    view.set_colormap("magma")

def test_micro_doppler_view(qapp):
    view = MicroDopplerView()
    # Data: [velocity, time] = [30, 40]
    data = np.random.rand(30, 40)
    payload = {
        "data": data,
        "vel_bins": np.linspace(-5, 5, 30),
        "time_bins": np.linspace(0, 2, 40)
    }
    view.set_data(payload)
    
    # Check image shape: should be [time, velocity] = [40, 30] due to transpose
    image_data = view.image.image
    assert image_data.shape == (40, 30)
    assert np.allclose(image_data, data.T)
    
    # Test dB mode
    view.set_db_mode(True)
    image_data_db = view.image.image
    expected_db = 20 * np.log10(np.maximum(np.abs(data.T), 1e-12))
    assert np.allclose(image_data_db, expected_db)

def test_doppler_azimuth_view(qapp):
    view = DopplerAzimuthView()
    # Data: [velocity, angle] = [15, 25]
    data = np.random.rand(15, 25)
    payload = {
        "data": data,
        "vel_bins": np.linspace(-5, 5, 15),
        "angle_bins": np.linspace(-1, 1, 25)
    }
    view.set_data(payload)
    
    # Check image shape: should be [angle, velocity] = [25, 15] due to transpose
    image_data = view.image.image
    assert image_data.shape == (25, 15)
    assert np.allclose(image_data, data.T)
    
    # Test dB mode
    view.set_db_mode(True)
    image_data_db = view.image.image
    expected_db = 20 * np.log10(np.maximum(np.abs(data.T), 1e-12))
    assert np.allclose(image_data_db, expected_db)

def test_range_doppler_view(qapp):
    view = RangeDopplerView()
    # Data: [range, velocity] = [50, 60]
    data = np.random.rand(50, 60)
    payload = {
        "data": data,
        "range_bins": np.linspace(0, 10, 50),
        "vel_bins": np.linspace(-5, 5, 60)
    }
    view.set_data(payload)
    
    # Check image shape: should be [velocity, range] = [60, 50] due to transpose
    image_data = view.image.image
    assert image_data.shape == (60, 50)
    assert np.allclose(image_data, data.T)
    
    # Test dB mode
    view.set_db_mode(True)
    image_data_db = view.image.image
    expected_db = 20 * np.log10(np.maximum(np.abs(data.T), 1e-12))
    assert np.allclose(image_data_db, expected_db)

def test_range_response_view(qapp):
    view = RangeResponseView()
    # Data: [range] = [100]
    data = np.random.rand(100)
    payload = {
        "data": data,
        "range_bins": np.linspace(0, 10, 100)
    }
    view.set_data(payload)
    
    # Check curve data
    x_data, y_data = view.curve.getData()
    assert x_data.shape == (100,)
    assert y_data.shape == (100,)
    assert np.allclose(y_data, np.abs(data)) # Default is mag
    
    # Test dB mode
    view.set_db_mode(True)
    _, y_data_db = view.curve.getData()
    expected_db = 20 * np.log10(np.maximum(np.abs(data), 1e-12))
    assert np.allclose(y_data_db, expected_db)

def test_range_doppler_detector_2d_view(qapp):
    from mmwave_radar_processing.visualization.views.range_doppler_detector_view import RangeDopplerDetectorView
    view = RangeDopplerDetectorView()
    # Data: [range, velocity] = [50, 60]
    data = np.random.rand(50, 60)
    # Detections: [range_idx, vel_idx]
    dets = np.array([[10, 20], [30, 40]])
    
    payload = {
        "data": dets, # Processor returns detections as primary data
        "rng_dop_resp": data, # Heatmap provided separately
        "range_bins": np.linspace(0, 10, 50),
        "vel_bins": np.linspace(-5, 5, 60),
        "dets": dets
    }
    view.set_data(payload)
    
    # Check heatmap image shape (should be [velocity, range] = [60, 50])
    image_data = view.image.image
    assert image_data.shape == (60, 50)
    assert np.allclose(image_data, data.T)
    
    # Check scatter plot data
    x_data = view.scatter.data['x']
    y_data = view.scatter.data['y']
    
    assert len(x_data) == 2
    assert len(y_data) == 2
    
    # Check values
    expected_x = payload["vel_bins"][dets[:, 1]]
    expected_y = payload["range_bins"][dets[:, 0]]
    
    assert np.allclose(x_data, expected_x)
    assert np.allclose(y_data, expected_y)

def test_range_detector_view(qapp):
    from mmwave_radar_processing.visualization.views.range_detector_view import RangeDetectorView
    view = RangeDetectorView()
    
    # Data: [range] = [100]
    data = np.random.rand(100)
    thresholds = np.random.rand(100) * 0.5
    dets = np.array([10, 50, 80])
    range_bins = np.linspace(0, 10, 100)
    
    payload = {
        "range_resp": data,
        "thresholds": thresholds,
        "dets": dets,
        "range_bins": range_bins
    }
    
    view.set_data(payload)
    
    # Check signal curve
    x_data, y_data = view.curve.getData()
    assert np.allclose(y_data, np.abs(data))
    
    # Check threshold curve
    x_thresh, y_thresh = view.threshold_curve.getData()
    assert np.allclose(y_thresh, thresholds)
    
    # Check detections scatter
    x_scat = view.scatter.data['x']
    y_scat = view.scatter.data['y']
    
    assert len(x_scat) == 3
    assert np.allclose(x_scat, range_bins[dets])
    assert np.allclose(y_scat, np.abs(data[dets]))

def test_point_cloud_view(qapp):
    from mmwave_radar_processing.visualization.views.point_cloud_view import PointCloudView
    view = PointCloudView()
    
    # Data: N x 4 (x, y, z, vel)
    data = np.random.rand(10, 4)
    payload = {"data": data}
    
    view.set_data(payload)
    
    # Check scatter plot data
    pos = view.scatter.pos
    assert pos.shape == (10, 3)
    assert np.allclose(pos, data[:, :3])
    
    # Check colors (should be set)
    color = view.scatter.color
    assert color.shape == (10, 4)
    
    # Check color bar existence
    assert hasattr(view, 'cb_widget')
    assert hasattr(view, 'cb_plot')
    assert hasattr(view, 'cb_img')
    
    # Check for ThickerAxisItem
    from mmwave_radar_processing.visualization.views.point_cloud_view import ThickerAxisItem
    found_axis = False
    for item in view.plot.items:
        if isinstance(item, ThickerAxisItem):
            found_axis = True
            break
    assert found_axis

def test_altitude_view(qapp):
    from mmwave_radar_processing.visualization.views.altitude_view import AltitudeView
    view = AltitudeView()
    
    # Data
    coarse_fft = np.random.rand(100)
    range_bins = np.linspace(0, 10, 100)
    altitude = 5.5
    
    payload = {
        "coarse_fft_data": coarse_fft,
        "range_bins": range_bins,
        "current_altitude_corrected_m": altitude
    }
    
    view.set_data(payload)
    
    # Check signal curve
    x_data, y_data = view.curve.getData()
    assert np.allclose(y_data, np.abs(coarse_fft))
    
    # Check altitude line
    assert view.altitude_line.value() == altitude
    assert view.altitude_line.isVisible()
    
    # Test hidden line for invalid altitude
    payload["current_altitude_corrected_m"] = -1.0
    view.set_data(payload)
    assert not view.altitude_line.isVisible()

def test_processor_view_panel_caching(qapp):
    from mmwave_radar_processing.visualization.gui.processor_view_panel import ProcessorViewPanel
    from mmwave_radar_processing.visualization.backends.processor_registry import get_default_registry
    
    registry = get_default_registry()
    panel = ProcessorViewPanel(registry)
    panel.show()
    
    # Simulate incoming data for a view that is NOT currently visible
    # micro_doppler_resp is not in the default 2x2 grid
    key = "micro_doppler_resp"
    payload = {
        "data": np.zeros((10, 10)),
        "time_bins": np.arange(10),
        "vel_bins": np.arange(10)
    }
    
    # Update panel (should cache but NOT update widget since it's hidden)
    panel.handle_view_update(key, payload)
    assert key in panel.latest_payloads
    assert panel.latest_payloads[key] is payload
    
    widget = panel.view_widgets[key]
    # Ensure it didn't get updated yet (last_payload is placeholder data, not our new payload)
    assert widget.last_payload is not payload
    
    # Now select the view in cell (0, 0)
    layout = panel.layout()
    item = layout.itemAtPosition(0, 0)
    cell_widget = item.widget()
    combo = cell_widget.findChild(QComboBox)
    
    # Set to the key
    index = combo.findData(key)
    combo.setCurrentIndex(index)
    
    # Check if the widget got the data
    assert widget.isVisible()
    # RangeDopplerView stores last_payload in BaseView
    assert widget.last_payload is payload
