import streamlit as st
from streamlit_webrtc import webrtc_streamer, ClientSettings, VideoTransformerBase
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import av
from yolov5.utils.plots import Annotator, colors

class YOLOv5VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_img = self.preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            results = self.model(input_img)

        labels = results.xyxy[0][:, -1].numpy()
        boxes = results.xyxy[0][:, :-1].numpy()

        annotator = Annotator(img_rgb)
        for i, (label, box) in enumerate(zip(labels, boxes)):
            color = colors(int(label))
            annotator.box_label(box, f"{label}: {box[4]:.2f}", color=color)

        result_img = cv2.cvtColor(annotator.im, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")


st.header("Object Detection with YOLOv5")
st.markdown("Click the 'Start' button below to access your webcam and see the object detection in real-time.")

webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=YOLOv5VideoTransformer)