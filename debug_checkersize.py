import cv2
import numpy as np
import time

def brute_force_checker_size(frame, max_rows=7, max_cols=7):
    """
    Brute-force mencari ukuran checkerboard dari (3x3) sampai (max_rows x max_cols).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing untuk membantu deteksi
    # gray = cv2.equalizeHist(gray)  # Kadang membantu, kadang tidak
    
    found_size = None
    best_corners = None
    
    # Loop semua ukuran (mulai dari yang besar ke kecil)
    for r in range(max_rows, 2, -1):
        for c in range(max_cols, 2, -1):
            
            # Coba ukuran (r, c)
            try:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                ret, corners = cv2.findChessboardCorners(gray, (c, r), flags)  # Note: (cols, rows)
                
                if ret:
                    # Double check kualitas corner (opsional)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    return (c, r), corners_sub
            except:
                continue
                
    return None, None

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Matikan autofocus jika bisa
    
    print("Mencari ukuran checkerboard...")
    print("Arahkan kamera ke board vertikal saja (tutup ChArUco jika perlu).")
    print("Tekan Q untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        
        # Cari ukuran
        size, corners = brute_force_checker_size(frame)
        
        if size:
            # Gambar jika ketemu
            cv2.drawChessboardCorners(display, size, corners, True)
            
            # Tulis ukuran di layar dengan JELAS
            text = f"FOUND: {size[0]} x {size[1]} inner corners"
            cv2.putText(display, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Juga print di terminal supaya user sadar
            # print(f"Found: {size}")
        else:
            cv2.putText(display, "Scanning...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Checkerboard Size Detector", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
