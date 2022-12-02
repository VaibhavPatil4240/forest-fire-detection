from openCVFireDetection import FireDetectionCV
import streamlit as st
import torch
import cv2
import numpy as np

@st.cache()
def load_model():
    model = torch.hub.load('ultralytics/yolov5','custom',path="weights/last.pt",force_reload=True)
    model.conf = 0.50
    return model

def main():
    st.title("Fire Detection App")
    model=load_model()
    cvFrame=st.empty()
    mlFrame=st.empty()
    fireDetector=FireDetectionCV()
    cam = cv2.VideoCapture(0)
    while(True):
        ret,frame=cam.read()
        result=fireDetector.detectFire(frame)
        if(result[0]):
            results = model(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            length = len(results.xyxy[0])

            output = np.squeeze(results.render())
            mlFrame.image(
                output,caption='Ml Model',
                width=600
                )
        cvFrame.image(
            [cv2.cvtColor(result[1],cv2.COLOR_HSV2RGB),cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)],
            caption=['Filter','Original Feed'],
            width=300)
        
if __name__=="__main__":
    main()