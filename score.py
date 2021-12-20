
import torch
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import json
import numpy as np
import cv2
# import tensorflow as tf


st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("AVA Product Finder")
st.text("provide image url")

@st.cache(allow_output_mutation=True)
def load_model():
# Model
#   model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
  model = torch.hub.load('', 'custom', source='local',force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom
  
  return model

with st.spinner("Loading model into memory--"):
  model=load_model()

def decode_img(image):
  print("--image--", image)
  # image_bytes=image.read()
  img = Image.open(BytesIO(image))

  results = model(img, size=640)  # reduce size=320 for faster inference
#   return results.pandas().xyxy[0].to_json(orient="records")
  return results.pandas().xyxy[0].to_json(orient="records")


path=st.text_input("enter image URL to detect--", 'https://ultralytics.com/images/zidane.jpg')

if path is not None:
  content=requests.get(path).content

  st.write("predictions:")
  image=Image.open(BytesIO(content))
  op1=np.array(image)
  
  color = (255, 0, 0)
  thickness = 2
  
  with st.spinner("--classifying--"):
    label=decode_img(content)
    label=json.loads(label)
    print("--label--", label)
    for t1 in label:
      print(t1)
      bboxes=[t1["xmin"],t1["ymin"],t1["xmax"],t1["ymax"]]
      label_id=t1["class"]
      class_name=t1["name"]
      conf_score=t1["confidence"]
      draw=cv2.rectangle(op1,(bboxes[0],bboxes[1]),(bboxes[2],bboxes[3]),color, thickness)
#       draw = ImageDraw.Draw(op1)
#       draw.rectangle([bboxes[0],bboxes[1],bboxes[2],bboxes[3]], width = 10, outline="#0000ff")
      
      
      
    
      st.write(draw)
  st.write("")
  st.image(draw, caption="predictions")


# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# print(results.print())  # or .show(), .save(), .crop(), .pandas(), etc.
