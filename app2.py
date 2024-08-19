import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from PIL import Image

# Function to analyze facial emotion using DeepFace
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result

def overlay_text_on_frame(frame, text):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return frame

def facesentiment(image):
    # Convert the image to BGR for OpenCV
    image_bgr = np.array(image.convert('RGB'))[:, :, ::-1].copy()

    # Analyze the frame using DeepFace
    result = analyze_frame(image_bgr)

    # Extract the face coordinates
    face_coordinates = result[0]["region"]
    x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

    # Draw bounding box around the face
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"Emotion: {result[0]['dominant_emotion']}"
    cv2.putText(image_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Convert the BGR frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Overlay text on the frame
    frame_with_overlay = overlay_text_on_frame(frame_rgb, text)

    return frame_with_overlay

def main():
    st.title("Image Emotion Detection")
    activities = ["Upload Image", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Karthik.M and Vishnu Anand TA  
            Email : karthikmandigiri@gmail.com  
        """)
    if choice == "Upload Image":
        st.subheader("Upload an Image for Emotion Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Detecting emotion...")
            frame_with_overlay = facesentiment(image)
            st.image(frame_with_overlay, caption='Processed Image', use_column_width=True)

    elif choice == "About":
        st.subheader("About this app")

        html_temp4 = """
                                     		<div style="background-color:#98AFC7;padding:10px">
                                     		<h4 style="color:white;text-align:center;">This Application is developed by M.Karthik and Vishnu Anand TA  </h4>
                                     		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                     		</div>
                                     		<br></br>
                                     		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass

if __name__ == "__main__":
    main()
