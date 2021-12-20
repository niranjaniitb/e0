# import cv2
# import streamlit as st

# st.title("Webcam Application")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# cam = cv2.VideoCapture(0)

# while run:
#     ret, frame = cam.read()
#     print("--", ret)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')
    
    
import streamlit as st
from webcam import webcam

captured_image = webcam()
if captured_image is None:
    st.write("Waiting for capture...")
else:
    st.write("Got an image from the webcam:")
    st.image(captured_image)
    

