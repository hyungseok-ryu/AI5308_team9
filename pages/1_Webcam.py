import streamlit as st
from streamlit_webrtc import webrtc_streamer, ClientSettings

# Function to display the webcam feed
def main():
    st.title("Real-time Webcam Video")
    
    st.markdown("## Webcam Video")
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=None)
    
    if webrtc_ctx.video_transformer:
        st.markdown("Stream is running")
    else:
        st.markdown("Starting stream...")


if __name__ == "__main__":
    main()