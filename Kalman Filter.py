import time
import numpy as np

class KalmanFilter:
    def __init__(self):
        #State Matrix: Inputted Values
        self.x = np.zeros(4) 

        #State Transition Matrix: How the system changes
        self.F = np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]])
        
        #Measurement Function: What values the sensor is reading 
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])
        
        self.P = np.eye(4) * 200        #Initial uncertainty
        self.R = np.eye(2) * 80        #Measurement Noise (Ignore Sudden Jumps)
        self.Q = np.eye(4) * 0.01      #Uncertainty in predictions.  

    def predict(self):

        self.x = self.F @ self.x                            #Predict new states
        self.P = self.F @ self.P @ self.F.T + self.Q        #Predict the updated uncertainty

    def update(self, measurement):
        z = np.array(measurement)

        
        y = z - (self.H @ self.x)            #Difference between actual measurement and expected measurement

        S = self.H @ self.P @ self.H.T + self.R     #Innovation Covariance: Uncertainty in prediction + measurement 
        
        # Added safety for singular matrix
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        
        self.x = self.x + (K @ y)   #Update state estimate

        I = np.eye(4)               #Identity Matrix

        self.P = (I - (K @ self.H)) @ self.P    #Updated Uncertainty


class KalmanTracker:
    def __init__(self, threshold = 5):
        self.k = KalmanFilter()         #Assign the Kalman filter
        self.initiated = False          #Assign obstalce as not detected initially
        self.hit_streak = 0             #Initialize hit streak tracker
        self.threshold = threshold      #Define the hit threshold
        self.last_time = time.time()    #Define the time variable used to calculate dt
        
    def update(self, raw_x, raw_y):
        current_time = time.time()

        #If obstacle is not detected
        if not self.initiated:
            self.k.x = np.array([float(raw_x), float(raw_y), 0, 0])       # Assign inputs to the matrix
            self.initiated = True
            self.hit_streak = 1
            return raw_x, raw_y
        
        dt = current_time - self.last_time
        self.last_time = current_time

        #Calculate dt
        self.k.F[0, 2] = dt
        self.k.F[1, 3] = dt

        self.k.predict()
        
        #Gating
        dist = np.sqrt((self.k.x[0] - raw_x)**2 + (self.k.x[1] - raw_y)**2)

        startup = self.hit_streak < self.threshold      #Gating is ignored on initial measurements due to big jumps being needed.

        if dist < 800 or startup:
            self.k.update([raw_x, raw_y])       #Call the update functions with the inputs
            self.hit_streak += 1                
        else:
            self.hit_streak = max(0, self.hit_streak - 1)       #Reduce hit streak by one instead of resetting so target isnt lost if 1 frame is gone 
        
        return self.k.x[0], self.k.x[1]

    @property
    def is_confirmed(self):
        return self.hit_streak >= self.threshold

    def reset(self):
        self.initiated = False
        self.hit_streak = 0
