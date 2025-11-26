import sys
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

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
