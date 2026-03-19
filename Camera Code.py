import time
import math
from ai_camera import IMX500Detector

#CONSTANT VARIABLES
width = 640
fov = 75
center = width/2
degree_per_pixel = fov/width

camera = IMX500Detector()
camera.start(show_preview = False)

try:
    while True:
        detections = camera.get_detections()
        labels = camera.get_labels()
        
        current_frame_data = []

        for det in detections:
            label = labels[int(det.category)]
            
            if det.conf > 0.5:
                x, y, w, h = det.box
                center_x = x + (w / 2)

                offset_from_center = center_x - center
                calculated_angle = offset_from_center * degree_per_pixel

                print(f"label: {label}, angle:{calculated_angle}, conf:{det.conf}")

        time.sleep(0.1)

except KeyboardInterrupt:
    camera.stop()
