from .camera import Camera
from .turntable import TurntableController
from .laser import LaserDetector
from .triangulation import TriangulationEngine
# from .auto_calibration import AutoCalibration # Not used in system init directly anymore?
# Actually app.py uses system.calibration which was the OLD calibration class.
# But system.auto_calib was the NEW one in Next.
# In Lite, there was 'calibration.py' which was unrelated to 'auto_calibration.py'.
# Let's import the new AutoCalibration for consistency if usage matches.
from .auto_calibration import AutoCalibration
from .scanner import Scanner
import threading
import time

class SystemController:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SystemController()
        return cls._instance

    def __init__(self):
        self.camera = Camera(index=0) # Default index
        
        # Hardware
        # Updated to new TurntableController (HTTP)
        self.turntable = TurntableController(ip_address="10.42.37.40")
        
        self.laser = LaserDetector()
        
        # Processing
        self.triangulation = TriangulationEngine()
        self.auto_calib = AutoCalibration()
        
        # Auto-load calibration if exists
        import os
        if os.path.exists("calibration.json"):
            self.triangulation.load_calibration("calibration.json") 
        
        # Logic
        self.scanner = Scanner(self)
        
        self.is_scanning = False
        self.is_calibrating = False
        
        # Legacy/Compatibility layer if needed for old app.py calls
        # app.py calls system.calibration.matrix... 
        # We need to ensure app.py is updated to use system.triangulation OR map it here.
        # For now, let's keep the structure clean and update app.py later.
        
    def get_status(self):
        # Sync turntable status if possible
        if self.turntable:
            self.turntable.sync_status()
        
        return {
            "camera": self.camera.cap is not None and self.camera.cap.isOpened() if self.camera.cap else False,
            "turntable": self.turntable.connected if self.turntable else False,
            "turntable_pos": self.turntable.current_steps if self.turntable else 0,
            "scanning": self.is_scanning
        }
