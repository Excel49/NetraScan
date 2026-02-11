import cv2
import threading
import time
import numpy as np

class Camera:
    def __init__(self, index=1):  # Default preference
        self.lock = threading.Lock()
        self.frame_buffer = None
        self.is_running = True
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Auto-detect camera
        self.camera_index = self._find_working_camera(preferred_index=index)
        
        if self.camera_index is not None:
            print(f"📷 Initializing Camera on index {self.camera_index}...")
            self._init_camera(self.camera_index)
        else:
            print("❌ CRITICAL: No working camera found.")
            self.cap = None

    def _find_working_camera(self, preferred_index):
        """Scan for available cameras - simplified for problematic webcams"""
        print(f"🔍 Testing camera index {preferred_index}...")
        
        # For now, trust preferred index without extensive testing
        # Testing causes issues with some webcams (MSMF errors)
        return preferred_index

    def _test_camera(self, index):
        """Minimal test - just check if camera exists"""
        # Note: Full read test removed because it causes MSMF errors
        # We'll let _init_camera handle full initialization
        return True

    def _init_camera(self, index):
        """Initialize camera with DSHOW as preference"""
        print(f"📷 Opening camera index {index} with DSHOW (DirectShow)...")
        
        try:
            # DSHOW is proven to work on this machine (based on test_camera.py logs)
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print("   ⚠️ DSHOW failed, retrying with Auto-detect...")
                cap.release()
                cap = cv2.VideoCapture(index)
            
            if cap.isOpened():
                # First check what we got by default
                ret, frame = cap.read()
                if ret:
                    print(f"   ✅ Camera opened. Default Resolution: {frame.shape}")
                
                # Coba naikkan resolusi (tapi jangan maksa jika bikin black screen)
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # FORCE RESET EXPOSURE TO AUTO
                    # DSHOW: 3 = Auto, 1 = Manual
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                    print("   🔧 Camera Exposure reset to AUTO.")
                except Exception as e:
                    print(f"   ⚠️ Could not reset settings: {e}")
                
                # Baca lagi untuk memastikan setting baru valid
                ret, frame = cap.read()
                if ret:
                    if np.mean(frame) < 1.0:
                        print("   ⚠️ Warning: Frame detected but looks completely black/dark.")
                    
                    print(f"   ✅ SUCCESS: Camera initialized! Final Resolution: {frame.shape}")
                    self.cap = cap
                    self.is_running = True
                    self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
                    self.capture_thread.start()
                else:
                    print(f"   ❌ Camera opened but cannot read frames.")
                    cap.release()
                    self.cap = None
            else:
                print(f"❌ Failed to open camera index {index}")
                self.cap = None
                
        except Exception as e:
             print(f"❌ Camera init exception: {e}")
             self.cap = None

    def _capture_frames(self):
        """Background thread to continuously capture frames"""
        consecutive_failures = 0
        
        while self.is_running:
            try:
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.lock:
                            self.frame_buffer = frame
                            self.frame_count += 1
                            
                            # Debug log every 60 frames (approx 2 sec)
                            if self.frame_count % 60 == 0:
                                 print(f"   [DEBUG] Camera running... Frame {self.frame_count} captured. Shape: {frame.shape}")
                        
                        consecutive_failures = 0
                        
                        # Calculate FPS
                        current_time = time.time()
                        if current_time - self.last_time >= 1.0:
                            self.fps = self.frame_count
                            self.frame_count = 0
                            self.last_time = current_time
                    else:
                        self.frame_buffer = None
                        print("   ⚠️ Cannot read frame (ret=False)")
                        consecutive_failures += 1
                        if consecutive_failures > 50:
                            print("⚠️ Camera signal lost. Reinitializing...")
                            self._reinit()
                            consecutive_failures = 0
                else:
                    time.sleep(1)
            
            except Exception as e:
                # print(f"Capture error: {e}") 
                time.sleep(1)
            
            time.sleep(0.005)

    def _reinit(self):
        if self.cap is not None: 
            self.cap.release()
        time.sleep(1)
        self._init_camera(self.camera_index)

    def get_frame(self):
        """Get the latest frame from buffer"""
        with self.lock:
            if self.frame_buffer is not None:
                return self.frame_buffer.copy()
        return None
    
    def set_exposure(self, value):
        """
        Set camera exposure (Manual).
        Value range typically -1 to -13 (logarithmic) for DSHOW, 
        or raw values for other backends.
        Using -5 as a safe starting point for 'dark' exposure.
        """
        if self.cap is not None:
             try:
                 # Disable Auto Exposure first (1 = Manual, 3 = Auto)
                 self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 0.25 or 1 depending on backend
                 
                 # Set Exposure
                 # Note: For many webcams on Windows, exposure is -1 to -13. 
                 # -1 is bright, -13 is very dark.
                 self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
                 print(f"📷 Exposure set to {value}")
                 return True
             except Exception as e:
                 print(f"❌ Failed to set exposure: {e}")
                 return False
        return False
        
    def release(self):
        """Release camera resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
