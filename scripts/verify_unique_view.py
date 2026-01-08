
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QComboBox
from PyQt6.QtCore import Qt, QTimer

# Add repo root to path
sys.path.append(os.getcwd())

from mmwave_radar_processing.visualization.gui.processor_view_panel import ProcessorViewPanel
from mmwave_radar_processing.visualization.backends.processor_registry import ProcessorSpec

# Mock View Class
class MockView(QWidget):
    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.label = QLabel("I am a plot", self)
        self.layout.addWidget(self.label)

    def set_data(self, payload):
        pass
        
    def set_db_mode(self, enabled):
        pass

def main():
    app = QApplication(sys.argv)
    
    # Create mock registry
    registry = {
        "view1": ProcessorSpec(key="view1", display_name="View 1", processor_cls=None, view_cls=MockView, required_inputs=None, output_schema=None),
        "view2": ProcessorSpec(key="view2", display_name="View 2", processor_cls=None, view_cls=MockView, required_inputs=None, output_schema=None),
        "range_doppler_resp": ProcessorSpec(key="range_doppler_resp", display_name="Range-Doppler", processor_cls=None, view_cls=MockView, required_inputs=None, output_schema=None),
        "range_angle_resp": ProcessorSpec(key="range_angle_resp", display_name="Range-Angle", processor_cls=None, view_cls=MockView, required_inputs=None, output_schema=None),
        "range_resp": ProcessorSpec(key="range_resp", display_name="Range Response", processor_cls=None, view_cls=MockView, required_inputs=None, output_schema=None),
        "doppler_azimuth_resp": ProcessorSpec(key="doppler_azimuth_resp", display_name="Doppler-Azimuth", processor_cls=None, view_cls=MockView, required_inputs=None, output_schema=None),
    }
    
    window = QMainWindow()
    panel = ProcessorViewPanel(registry, parent=window)
    window.setCentralWidget(panel)
    window.resize(800, 600)
    window.show()
    
    print("Window shown. Starting verification...")
    
    def verify_logic():
        # Initial state: (0,0) is range_doppler_resp
        # Let's try to set (0,1) to range_doppler_resp
        # This should clear (0,0)
        
        print("Step 1: Setting (0,1) to 'range_doppler_resp' (duplicate of (0,0))")
        
        # Find combo for (0,1)
        combo_01 = panel.layout().itemAtPosition(0, 1).widget().findChild(QComboBox)
        index = combo_01.findData("range_doppler_resp")
        combo_01.setCurrentIndex(index)
        
        # Wait and check
        QTimer.singleShot(500, check_step_1)
        
    def check_step_1():
        # Check if (0,0) is now None
        combo_00 = panel.layout().itemAtPosition(0, 0).widget().findChild(QComboBox)
        current_data_00 = combo_00.currentData()
        
        print(f"Step 1 Result: (0,0) current data: {current_data_00}")
        
        if current_data_00 is None:
            print("SUCCESS: (0,0) was cleared!")
        else:
            print(f"FAILURE: (0,0) is still {current_data_00}")
            
        # Check if (0,1) is range_doppler_resp
        combo_01 = panel.layout().itemAtPosition(0, 1).widget().findChild(QComboBox)
        current_data_01 = combo_01.currentData()
        print(f"Step 1 Result: (0,1) current data: {current_data_01}")
        
        if current_data_01 == "range_doppler_resp":
             print("SUCCESS: (0,1) is set correctly!")
        else:
             print(f"FAILURE: (0,1) is {current_data_01}")

        app.quit()

    QTimer.singleShot(1000, verify_logic)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
