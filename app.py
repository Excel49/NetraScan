import sys
import os

# Ensure the current directory is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from flask import Flask, render_template, Response, jsonify, request
    from flask_socketio import SocketIO
    from core.system import SystemController
    import cv2
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you have installed requirements: pip install -r requirements.txt")
    sys.exit(1)

import time
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'netrascan-lite-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

system = SystemController.get_instance()

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

from core.utils import get_laser_mask
import core.config as config

@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'raw') # 'raw' or 'laser'
    system = SystemController.get_instance()
    
    def generate():
        while True:
            frame = system.camera.get_frame()
            if frame is not None:
                display_frame = frame.copy()
                
                if mode == 'laser':
                    # Use shared utility for consistency
                    # Use current dynamic threshold if set, or default
                    current_thresh = system.laser.threshold if system.laser else config.LASER_THRESHOLD
                    mask = get_laser_mask(frame, custom_threshold=current_thresh)
                    
                    # Create black background
                    display_frame[:] = 0 
                    
                    # Apply mask: Set valid laser pixels to GREEN
                    # Apply mask: Set valid laser pixels to GREEN
                    display_frame[mask > 0] = (0, 255, 0)

                elif mode == 'axis':
                    # Draw World Origin Axes - detect ChArUco in real-time
                    if system.triangulation.calibrated:
                        import numpy as np
                        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                        corners, ids, _, _ = system.auto_calib.charuco_detector.detectBoard(gray)
                        
                        if corners is not None and len(corners) > 4:
                            # Real-time pose estimation
                            all_board_corners = system.auto_calib.charuco_board.getChessboardCorners()
                            obj_pts = []
                            img_pts = []
                            
                            for j, cid in enumerate(ids):
                                id_val = int(cid[0])
                                if id_val < len(all_board_corners):
                                    obj_pts.append(all_board_corners[id_val])
                                    img_pts.append(corners[j])
                            
                            if len(obj_pts) >= 4:
                                obj_pts_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
                                img_pts_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 2)
                                
                                # Offset object points to center the origin
                                # Board is 5x5 squares * 0.025m = 0.125m
                                # Center offset = 0.0625m
                                center_offset = np.array([0.0625, 0.0625, 0], dtype=np.float32)
                                obj_pts_centered = obj_pts_arr - center_offset
                                
                                success, rvec, tvec = cv2.solvePnP(
                                    obj_pts_centered, img_pts_arr,
                                    system.triangulation.camera_matrix,
                                    system.triangulation.dist_coeffs
                                )
                                
                                if success:
                                    # Apply same axis mapping as point cloud
                                    # POINT_CLOUD_AXIS_MAP = [1, 2, 0]
                                    # New X = Old Y, New Y = Old Z, New Z = Old X
                                    from core.config import POINT_CLOUD_AXIS_MAP, POINT_CLOUD_AXIS_INVERT
                                    
                                    axis_len = 0.05
                                    # Original axes in board coords
                                    orig_axes = np.float32([
                                        [axis_len, 0, 0],     # Original X
                                        [0, axis_len, 0],     # Original Y 
                                        [0, 0, axis_len]      # Original Z
                                    ])
                                    
                                    # Apply axis map: [1,2,0] means X'=Y, Y'=Z, Z'=X
                                    mapped_x = orig_axes[POINT_CLOUD_AXIS_MAP[0]] * POINT_CLOUD_AXIS_INVERT[0]
                                    mapped_y = orig_axes[POINT_CLOUD_AXIS_MAP[1]] * POINT_CLOUD_AXIS_INVERT[1]
                                    mapped_z = orig_axes[POINT_CLOUD_AXIS_MAP[2]] * POINT_CLOUD_AXIS_INVERT[2]
                                    
                                    axis_pts = np.float32([
                                        [0, 0, 0],    # Origin
                                        mapped_x,     # X (Red)
                                        mapped_y,     # Y (Green)
                                        mapped_z      # Z (Blue)
                                    ])
                                    
                                    img_pts_axis, _ = cv2.projectPoints(
                                        axis_pts, rvec, tvec,
                                        system.triangulation.camera_matrix,
                                        system.triangulation.dist_coeffs
                                    )
                                    img_pts_axis = np.int32(img_pts_axis).reshape(-1, 2)
                                    
                                    origin = tuple(img_pts_axis[0])
                                    # X = Red, Y = Green, Z = Blue
                                    cv2.line(display_frame, origin, tuple(img_pts_axis[1]), (0, 0, 255), 3)
                                    cv2.line(display_frame, origin, tuple(img_pts_axis[2]), (0, 255, 0), 3)
                                    cv2.line(display_frame, origin, tuple(img_pts_axis[3]), (255, 0, 0), 3)
                                    
                                    # Draw origin point
                                    cv2.circle(display_frame, origin, 5, (255, 255, 255), -1)
                        else:
                            # No board detected - show text
                            cv2.putText(display_frame, "Place ChArUco board in view", (10, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        # Warning if not calibrated
                        cv2.putText(display_frame, "Not Calibrated", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Encode
                ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/scan/start', methods=['POST'])
def api_scan_start():
    if system.is_scanning:
        return jsonify({"success": False, "error": "Already scanning"}), 409
        
    threading.Thread(target=run_scan_thread, daemon=True).start()
    return jsonify({"success": True, "message": "Scan started"})

def run_scan_thread():
    """Thread to run the scan using Scanner class"""
    def progress_cb(step, total, points_count):
        socketio.emit('scan_progress', {
            'progress': ((step+1)/total)*100,
            'points_count': points_count,
            'current_angle': step
        })
        
    def data_cb(points_batch):
        socketio.emit('pointcloud_data', {'points': points_batch})

    try:
        points = system.scanner.start_scan(callback_progress=progress_cb, callback_data=data_cb)
        socketio.emit('scan_complete', {'status': 'completed', 'points_count': len(points)})
    except Exception as e:
        print(f"❌ Scan Error: {e}")
        socketio.emit('scan_complete', {'status': 'error', 'message': str(e)})

@app.route('/api/scan/stop', methods=['POST'])
def api_scan_stop():
    system.is_scanning = False
    return jsonify({"success": True, "message": "Scan stopped"})

@app.route('/api/settings/threshold', methods=['POST'])
def api_set_threshold():
    data = request.json
    try:
        val = int(data.get('threshold', 60))
        if system.laser:
            system.laser.threshold = val
            print(f"🔧 Laser Threshold updated to: {val}")
        return jsonify({"success": True, "threshold": val})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/settings/exposure', methods=['POST'])
def api_set_exposure():
    data = request.json
    try:
        val = int(data.get('exposure', -5))
        success = system.camera.set_exposure(val)
        if success:
             return jsonify({"success": True, "exposure": val})
        else:
             return jsonify({"success": False, "error": "Failed to set exposure"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/calibrate/camera', methods=['POST'])
def api_calibrate_camera():
    if system.is_scanning or system.is_calibrating:
        return jsonify({"success": False, "error": "System busy"}), 409
    threading.Thread(target=run_camera_calibration, daemon=True).start()
    return jsonify({"success": True, "message": "Camera Calibration started"})

@app.route('/api/calibrate/laser', methods=['POST'])
def api_calibrate_laser():
    if system.is_scanning or system.is_calibrating:
        return jsonify({"success": False, "error": "System busy"}), 409
    threading.Thread(target=run_laser_calibration, daemon=True).start()
    return jsonify({"success": True, "message": "Laser Calibration started"})

# Backward compatibility for existing UI button
@app.route('/api/calibrate', methods=['POST'])
def api_calibrate_legacy():
    return api_calibrate_camera()

@app.route('/api/config', methods=['GET'])
def api_get_config():
    """Return frontend configuration"""
    return jsonify({
        "camera_position": getattr(config, 'VISUALIZER_CAMERA_POSITION', [0, 8, 10]),
        # Add other frontend configs here if needed
    })

def capture_calibration_frames(num_captures=10, event_name='calibration_status'):
    """Helper to capture frames with turntable rotation"""
    frames = []
    
    # Determine steps
    steps_per_rev = config.TURNTABLE_STEPS_PER_REV
    if system.turntable and hasattr(system.turntable, 'steps_per_revolution'):
        steps_per_rev = system.turntable.steps_per_revolution
    
    steps_per_capture = int(steps_per_rev / num_captures)
    
    if system.turntable and system.turntable.connected:
        system.turntable.home()
        time.sleep(2)
        
    for i in range(num_captures):
        socketio.emit(event_name, {
            'message': f'Capturing image {i+1}/{num_captures}...', 
            'progress': int((i / num_captures) * 50)
        })
        
        # Rotate
        if system.turntable and system.turntable.connected:
            system.turntable.rotate(steps_per_capture)
            time.sleep(1.5)
        else:
            time.sleep(3.0) 
            
        frame = system.camera.get_frame()
        if frame is not None:
             frames.append(frame.copy())
             print(f"   Frame {i+1} captured")
             
    return frames

def run_camera_calibration():
    print("📷 Camera Calibration Thread Started")
    system.is_calibrating = True
    socketio.emit('calibration_status', {'message': 'Starting Camera Calibration...', 'progress': 0})
    
    try:
        frames = capture_calibration_frames(num_captures=12, event_name='calibration_status')
        
        socketio.emit('calibration_status', {'message': 'Processing images...', 'progress': 50})
        
        def progress_cb(pct):
             socketio.emit('calibration_status', {'message': f'Processing: {int(pct)}%', 'progress': 50 + (pct/2)})
             
        # Camera Calibration
        mtx, dist = system.auto_calib.calibrate_camera_charuco(frames)
        
        if mtx is not None:
            # Update Triangulation Engine
            system.triangulation.camera_matrix = mtx
            system.triangulation.dist_coeffs = dist
            system.triangulation.calibrated = True
            system.triangulation.save_calibration("calibration.json")
            
            socketio.emit('calibration_status', {
                'message': f'Cam Calib Success! Saved.', 
                'progress': 100, 
                'status': 'completed'
            })
        else:
            socketio.emit('calibration_status', {'message': 'Calibration Failed (No board detected).', 'status': 'error'})
            
    except Exception as e:
        print(f"❌ Calib Error: {e}")
        socketio.emit('calibration_status', {'message': f'Error: {str(e)}', 'status': 'error'})
    finally:
        system.is_calibrating = False

def run_laser_calibration():
    # Placeholder: Laser calibration usually requires specific target or manual input
    # For now, we reuse the manual calibration script logic or just skip if using auto-fixed
    # The new mechanism uses a fixed laser plane or separate manual calib tool
    print("🔴 Laser Calibration - Not fully implemented in Lite Auto Mode")
    socketio.emit('calibration_status', {'message': 'Laser Calib not available in Lite. Use defaults.', 'status': 'error'})
    system.is_calibrating = False

if __name__ == '__main__':
    print("🚀 Starting NetraScan-Lite Server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
