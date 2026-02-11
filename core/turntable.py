import requests
import time
import threading
import json
import core.config as config

class TurntableController:
    """
    Controller for ESP32 Turntable using HTTP Protocol
    Firmware: kodingan_esp (AccelStepper + WebServer)
    """
    
    def __init__(self, ip_address="192.168.1.39"):
        self.ip = ip_address
        self.base_url = f"http://{self.ip}"
        self.connected = False
        self.current_angle = 0.0
        self.steps_per_rev = config.TURNTABLE_STEPS_PER_REV
        self.steps_per_revolution = self.steps_per_rev # Compatibility alias
        self.steps_per_degree = self.steps_per_rev / 360.0
        
        # Status
        self.is_moving = False
        self.current_steps = 0
        
        # Background status polling
        self._monitor_thread = None
        self._stop_monitor = False
        
        print(f"🔄 Initializing Turntable at {self.base_url}...")
        # Check connection immediately
        self.connect()

    def connect(self):
        """Check connection to turntable"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.connected = True
                self.current_steps = int(data.get("current_position", 0))
                self.current_angle = (self.current_steps / self.steps_per_rev) * 360.0
                self.is_moving = data.get("is_moving", False)
                print(f"✅ Turntable connected at {self.ip}")
                return True
            else:
                self.connected = False
                return False
        except Exception as e:
            print(f"⚠️ Turntable connection failed: {e}")
            self.connected = False
            return False

    def rotate(self, steps, speed_profile=0):
        """
        Rotate by raw steps (Compatibility method)
        """
        if not self.connected and not self.connect():
            return False

        try:
            url = f"{self.base_url}/rotate"
            params = {"step": int(steps), "speed": speed_profile}
            
            print(f"🔄 Sending rotation command: {steps} steps...")
            response = requests.get(url, params=params, timeout=10) 
            
            if response.status_code == 200:
                data = response.json()
                self.current_steps = data.get("position", self.current_steps + int(steps))
                self.current_angle = (self.current_steps / self.steps_per_rev) * 360.0
                return True
            else:
                print(f"❌ Rotation failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Rotation error: {e}")
            return False

    def rotate_by(self, angle, speed_profile=0):
        """
        Rotate relative to current position
        angle: Degrees to rotate (can be negative)
        speed_profile: 0=Fast, 1=Medium, 2=Slow
        """
        if not self.connected and not self.connect():
            print("❌ Cannot rotate: Turntable not connected")
            return False

        steps = int(angle * self.steps_per_degree)
        
        try:
            url = f"{self.base_url}/rotate"
            params = {"step": steps, "speed": speed_profile}
            
            print(f"🔄 Sending rotation command: {angle} deg ({steps} steps)...")
            # Note: The firmware blocks for short moves, or we can use async requests
            # For simplicity in scanning loop, blocking is okay as long as timeout > move time
            response = requests.get(url, params=params, timeout=10) 
            
            if response.status_code == 200:
                data = response.json()
                self.current_steps = data.get("position", self.current_steps + steps)
                self.current_angle += angle
                return True
            else:
                print(f"❌ Rotation failed: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("⚠️ Rotation request timed out (motor might still be moving)")
            return True # Assume it's working
        except Exception as e:
            print(f"❌ Rotation error: {e}")
            return False

    def rotate_to(self, angle):
        """Rotate to absolute angle"""
        # Not fully implemented in firmware API comfortably without tracking revs
        # But we can calculate delta
        delta = angle - self.current_angle
        return self.rotate_by(delta)

    def home(self):
        """Home the turntable (set current pos to 0)"""
        if not self.connected and not self.connect():
            return False
            
        try:
            print("🏠 Homing turntable...")
            response = requests.get(f"{self.base_url}/home", timeout=30)
            if response.status_code == 200:
                self.current_angle = 0.0
                self.current_steps = 0
                print("✅ Homing complete")
                return True
            return False
        except Exception as e:
            print(f"❌ Homing error: {e}")
            return False

    def stop(self):
        """Emergency stop"""
        try:
            requests.get(f"{self.base_url}/estop", timeout=1)
            self.is_moving = False
            return True
        except:
            return False

    def get_status(self):
        """Get full status"""
        if not self.connect():
            return None
            
        return {
            "connected": self.connected,
            "angle": self.current_angle,
            "is_moving": self.is_moving,
            "steps": self.current_steps
        }

    def sync_status(self):
        """Update local state from hardware"""
        try:
            if self.connected:
                response = requests.get(f"{self.base_url}/status", timeout=0.5)
                if response.status_code == 200:
                    data = response.json()
                    self.current_steps = data.get("current_position", 0)
                    self.is_moving = data.get("is_moving", False)
                    # Recalculate angle based on steps?
                    # self.current_angle = (self.current_steps / self.steps_per_rev) * 360.0
        except:
            pass
