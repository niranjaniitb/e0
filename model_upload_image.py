
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

#-------

@st.cache(allow_output_mutation=True)
def load_model():
# Model
#   model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
  model = torch.hub.load('', 'custom', source='local',force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom
  
  return model

with st.spinner("Loading model into memory--"):
  model=load_model()

def decode_img(image):
#   print("--image--", image)
#   img = Image.open(BytesIO(image))
  st.write("model --input image shape--",image.shape)

  results = model(image) 
#   st.write("detection...")
#   st.write("detection...")
  # reduce size=320 for faster inference
  return results.pandas().xyxy[0].to_json(orient="records")


# path=st.text_input("enter image URL to detect--", 'https://ultralytics.com/images/zidane.jpg')


# if path is not None:
#   content=requests.get(path).content

st.title("Upload + Detection Example")
#---------
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("initial image type---",type(image))
    st.write("detection...")
#   st.write("predictions:")
#     image=Image.open(BytesIO(image))
    op1=np.array(image)
#     st.write("--image type after converting to npar----",type(op1), "--np shape--", op1.shape)
  

    color = (255,0,0)
    thickness = 2
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    f_color=(255,0,0)
    op_names={}

    with st.spinner("--classifying--"):
      label=decode_img(op1)
      label=json.loads(label)

      for t1 in label:
        temp=[]
        bboxes=[int(t1["xmin"]),int(t1["ymin"]),int(t1["xmax"]),int(t1["ymax"])]

        start=(bboxes[0],bboxes[1])
        end=(bboxes[2],bboxes[3])
        org = (int(bboxes[0]+(bboxes[2]-bboxes[0])/2),int(bboxes[1]+(bboxes[3]-bboxes[1])/2))

        label_id=t1["class"]
        class_name=t1["name"]
        conf_score=t1["confidence"]

        if label_id in op_names:
          op_names[label_id]= [op_names[label_id][0]+1,class_name]
        else:
          op_names[label_id]=[1,class_name]     

        op1=cv2.rectangle(op1,start,end,color, thickness)
        op1 = cv2.putText(op1, str(label_id), org, font,fontScale, f_color, thickness, cv2.LINE_AA)


    PIL_image = Image.fromarray(op1.astype('uint8'), 'RGB')

    st.write(op_names)
    st.write("")
    st.image(PIL_image, caption="predictions")

