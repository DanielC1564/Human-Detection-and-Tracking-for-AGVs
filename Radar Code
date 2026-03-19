import time
from rd03d import RD03D
from Kalman import KalmanFilter, KalmanTracker

#Initialize radar
radar = RD03D()
radar.set_multi_mode(False)   

t = {1: KalmanTracker(), 2: KalmanTracker(), 3: KalmanTracker()}

try:
    while True:
        if radar.update():
            for i in range(1, 4):
                target = radar.get_target(i)
                if target.distance > 0:
                    if 350 < target.distance < 1500:
                        x, y = t[i].update(target.x, target.y)
                        if t[i].is_confirmed:
                            print(f"CONFIRMED Target:{i} | X={int(x)}, Y={int(y)}")
                        else:
                            print(f"Tracking ID:{i}... (Confidence: {t[i].hit_streak}/{t[i].threshold})")
                    else:
                        t[i].hit_streak = max(0, t[i].hit_streak - 1)
                else:
                    t[i].reset()

                    
        time.sleep(0.1)

except KeyboardInterrupt:
	print("Stopping")
