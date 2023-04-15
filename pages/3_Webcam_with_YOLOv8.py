import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import av
import pandas
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors

class YOLOv8VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.names = self.model.names

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pil_img = Image.fromarray(img_rgb)
        # input_img = self.preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            results = self.model(img_rgb)

        labels = results[0].boxes.cls.numpy()
        boxes = results[0].boxes.xyxy.numpy()
        confidences = results[0].boxes.conf
        
        annotator = Annotator(img_rgb)
        for i, (label, box, confidence) in enumerate(zip(labels, boxes, confidences)):
            class_name = self.names[int(label)] 
            color = colors(int(label))
            annotator.box_label(box, f"{class_name}: {confidence:.2f}", color=color)
        result_img = cv2.cvtColor(annotator.im, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")


st.header("Object Detection with YOLOv8")
st.markdown("Click the 'Start' button below to access your webcam and see the object detection in real-time.")

webrtc_ctx = webrtc_streamer(key="YOLOv8",
                             mode=WebRtcMode.SENDRECV,
                             video_transformer_factory=YOLOv8VideoTransformer,
                             media_stream_constraints={"video": True, "audio": False},
                             async_processing=True,)