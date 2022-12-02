import cv2
import numpy as np
 
class FireDetectionCV:
    def __init__(self) -> None:
        self.lower = [18, 50, 50]
        self.upper = [35, 255, 255]
        self.lower = np.array(self.lower, dtype="uint8")
        self.upper = np.array(self.upper, dtype="uint8")
        
    def detectFire(self,frame)->bool:
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        output = cv2.bitwise_and(frame,hsv, mask=mask)
        no_red = cv2.countNonZero(mask)
        if int(no_red) > 2000:
            return True,output
        return False,output