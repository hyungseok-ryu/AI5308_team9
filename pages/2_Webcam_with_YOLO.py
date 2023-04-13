import streamlit as st
from streamlit_webrtc import webrtc_streamer, ClientSettings, VideoTransformerBase
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class YOLOv5VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_img = self.preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            results = self.model(input_img)

        result_img = results.render()[0]
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        return result_img

def main():
    st.title("YOLOv5 Object Detection with Webcam")
    st.markdown("## Webcam Video")

    yolo_transformer = YOLOv5VideoTransformer()
    
    if "_components_callbacks" not in st.session_state:
        st.session_state._components_callbacks = {}
        
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=yolo_transformer)

    if webrtc_ctx.video_transformer:
        st.markdown("Stream is running")
    else:
        st.markdown("Starting stream...")

if __name__ == "__main__":
    main()