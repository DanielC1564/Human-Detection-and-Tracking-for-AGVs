import threading
import time
import math
import numpy as np
from rd03d import RD03D
from ai_camera import IMX500Detector
from Kalman import KalmanFilter, KalmanTracker

# ------------------ SHARED VARIABLES --------------------
t_radar = {1: {'valid': False, 'x': 0, 'y': 0, 'dist': 0, 'angle': 0, 'time': 0},
           2: {'valid': False, 'x': 0, 'y': 0, 'dist': 0, 'angle': 0, 'time': 0},
           3: {'valid': False, 'x': 0, 'y': 0, 'dist': 0, 'angle': 0, 'time': 0}}

t_camera = []
camera_lock = threading.Lock()
running = True

# ------------------ RADAR THREAD ---------------------
def radar_collect():
    global t_radar
    radar = RD03D()
    radar.set_multi_mode(True)
    print("[Thread] Radar Started")    
    
    try:
        while running:
            if radar.update():
                for i in range(1, 4):
                    target = radar.get_target(i)
                    if target and 350 < target.distance < 7000:
                        t_radar[i]['valid'] = True
                        t_radar[i]['x'] = target.x
                        t_radar[i]['y'] = target.y
                        t_radar[i]['dist'] = target.distance
                        
                        # Calculate Angle for matching
                        if target.y > 0:
                            rad_angle = math.atan(target.x / target.y)
                            t_radar[i]['angle'] = math.degrees(rad_angle)
                        
                        t_radar[i]['time'] = time.time()
                    else:
                        t_radar[i]['valid'] = False
            time.sleep(0.1)
    finally:
        # Ensures radar closes even if thread crashes
        radar.close()
        print("[Thread] Radar Stopped")

# ------------------ CAMERA THREAD -----------------
def camera_collect():
    global t_camera
    
    #CONSTANTS
    WIDTH = 640
    FOV = 75
    CENTER = WIDTH / 2
    DEG_PER_PIXEL = FOV / WIDTH
    
    camera = None
    
    try:
        print("[Thread] Loading Camera Model... (This takes time)")
        camera = IMX500Detector()
        camera.start(show_preview=False)
        print("[Thread] Camera Running")

        while running:
            detections = camera.get_detections()
            labels = camera.get_labels()

            current_frame_data = []

            for det in detections:
                label = labels[int(det.category)]
                
                if det.conf > 0.5:
                    x, y, w, h = det.box
                    center_x = x + (w / 2)

                    #Pixel to Angle Math
                    offset_from_center = center_x - CENTER
                    calculated_angle = offset_from_center * DEG_PER_PIXEL

                    current_frame_data.append({
                        'label': label, 
                        'angle': calculated_angle, 
                        'conf': det.conf
                    })
            
            with camera_lock:
                t_camera = current_frame_data
            
            time.sleep(0.1)

    #Ensure Camera Closes properly to prevent Unicam Error
    finally:
        if camera:
            camera.stop()
            print("[Thread] Camera Released Safely")

# ------------------ MATCHING FUNCTION ------------------
def camera_radar_match(angle, detection_list):
    best_match = None
    max_diff = 20 

    for d in detection_list:
        if d['label'] != "person":
            continue
        diff = abs(angle - d['angle'])
        if diff < max_diff:
            max_diff = diff
            best_match = d
    
    return best_match

# ------------------ MAIN FUNCTION ----------------
if __name__ == "__main__":
    t1 = threading.Thread(target=radar_collect)
    t2 = threading.Thread(target=camera_collect)
    t1.start()
    t2.start()

    trackers = {1: KalmanTracker(), 2: KalmanTracker(), 3: KalmanTracker()}
    
    #Initialize last know y variable
    last_known_y = {1: 4000, 2: 4000, 3: 4000}    

    print("System Starting... (Press Ctrl+C to Stop)")

    try:
        while True:
            current_time = time.time()
            
            with camera_lock:
                snapshot = list(t_camera)

            for i in range(1, 4):
                radar_recent = (current_time - t_radar[i]['time']) < 0.5 and t_radar[i]['valid']
                
                #Reset raw coordinates to prevent bleedover from previous targets
                raw_x = 0
                raw_y = 0
                
                #Default to False. Only set True if data found.
                has_input = False 

                # --------- SENSOR FUSION LOGIC ------------
                if radar_recent:
                    raw_y = t_radar[i]['y']
                    title = "person"
                    
                    
                    last_known_y[i] = raw_y 

                    camera_match = camera_radar_match(t_radar[i]['angle'], snapshot)
                    
                    if camera_match:
                        # FUSION: Radar Depth + Camera Angle
                        title = camera_match['label']
                        angle_rad = math.radians(camera_match['angle'])
                        raw_x = raw_y * math.tan(angle_rad)
                    else:
                        # FALLBACK: Radar Depth + Radar Angle
                        raw_x = t_radar[i]['x']
                    
                    has_input = True
                
                # RADAR BLIND (Camera Only Mode)
                else:
                    #If Tracker 1 is blind, take the first person seen
                    if i == 1 and len(snapshot) > 0:
                        camera_match = snapshot[0]

                        title = camera_match['label']
                        # Use memory for depth
                        raw_y = last_known_y[i] 

                        angle_rad = math.radians(camera_match['angle'])
                        raw_x = raw_y * math.tan(angle_rad)
                        has_input = True

                # ------- UPDATE TRACKER --------------
                if has_input:
                    print(f"RAW DATA: X = {int(raw_x)},Y = {int(raw_y)}, OBS:{i}")
                    smooth_x, smooth_y = trackers[i].update(raw_x, raw_y)
                    
                    if trackers[i].is_confirmed:
                        if title == "person":
                            print(f"Obstacle {i}:PERSON DETECTED; X={int(smooth_x):4d}, Y={int(smooth_y):4d}")
                        else:
                            print(f"Obstacle {i}:INANIMATE DETECTED; X={int(smooth_x):4d}, Y={int(smooth_y):4d}")
                else:
                    # If no data, reset logic
                    trackers[i].reset()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping threads...")
        running = False
        t1.join()
        t2.join()
        print("System Shutdown Complete")
