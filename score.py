
import torch
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
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
  return results.pandas().xyxy[0].to_json(orient="records")

path=st.text_input("enter image URL to detect--", 'https://ultralytics.com/images/zidane.jpg')

if path is not None:
  content=requests.get(path).content

  st.write("predictions:")
  with st.spinner("--classifying--"):
    label=decode_img(content)
    st.write(label)
  st.write("")
  image=Image.open(BytesIO(content))
  st.image(image, caption="predictions")


# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# print(results.print())  # or .show(), .save(), .crop(), .pandas(), etc.
