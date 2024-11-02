import depthai as dai
import streamlit as st
import cv2
import numpy as np
import serial

# Function to initialize the Arduino connection
def init_arduino_connection():
    try:
        arduino = serial.Serial('COM3', 9600, timeout=1)
        st.session_state['arduino'] = arduino  # Store in session state
        st.success("Arduino connected!")
    except serial.SerialException as e:
        st.error(f"Error connecting to Arduino: {e}")

# Check if the Arduino connection exists in session state, if not, initialize it
if 'arduino' not in st.session_state:
    init_arduino_connection()

# Function to send commands to Arduino
def send_command(command):
    try:
        arduino = st.session_state['arduino']  # Get the stored arduino object
        arduino.write(command.encode())
        st.success(f"Sent {command} to Arduino!")
    except KeyError:
        st.error("Arduino not connected.")
    except serial.SerialException as e:
        st.error(f"Error sending command to Arduino: {e}")

# Create pipeline for DepthAI
pipeline = dai.Pipeline()

# Define sources and outputs for video
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 640)  # Match the input size of your model
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Create output for video stream
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.preview.link(xout_video.input)

# Load the YOLO model (from Roboflow, for example)
nn = pipeline.createYoloDetectionNetwork()
nn.setBlobPath("C:/Users/joyta/OneDrive/Desktop/Joy/best_openvino_2022.1_6shave.blob")  # Use your .blob file
nn.setConfidenceThreshold(0.5)  # Set your desired confidence threshold

# Set anchors and anchor masks
nn.setNumClasses(1)  # Adjust based on your model
nn.setCoordinateSize(4)
nn.setAnchors([])  # Update with your model's anchors
nn.setAnchorMasks({})  # Update with your model's anchor masks
nn.setIouThreshold(0.5)

# Link the camera output to the neural network
cam_rgb.preview.link(nn.input)

# Create output for detection results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    # Start a Streamlit web app
    st.title("DepthAI Live Feed with YOLOv5/ONNX Object Detection & Robot Control")

    # Get video stream from DepthAI
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    
    # Get neural network output queue (detection results)
    nn_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # Empty Streamlit placeholder for displaying frames
    frame_placeholder = st.empty()

    # Create buttons for controlling the robot
    st.subheader("Robot Controls")
    if st.button('Move Forward'):
        send_command('F')
    
    if st.button('Move Backward'):
        send_command('B')
    
    if st.button('Move Left'):
        send_command('L')
    
    if st.button('Move Right'):
        send_command('R')
    
    if st.button('Stop'):
        send_command('S')

    # Create a stop button for the webcam
    stop_button_pressed = st.button("Stop Webcam")

    while not stop_button_pressed:
        # Get frame from DepthAI
        in_video = video_queue.get()
        frame = in_video.getCvFrame()

        if frame is None:
            st.error("Failed to grab frame.")
            break

        # Get detection results from the neural network
        in_nn = nn_queue.get()
        if in_nn is not None:
            detections = in_nn.detections  # Retrieve detections

            # Draw bounding boxes and labels on the frame
            for detection in detections:
                x1, y1, x2, y2 = int(detection.xmin * frame.shape[1]), int(detection.ymin * frame.shape[0]), \
                                 int(detection.xmax * frame.shape[1]), int(detection.ymax * frame.shape[0])
                label = f"Class: Person, Conf: {detection.confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert BGR to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the frame in the placeholder
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

# Cleanup resources after stopping
cv2.destroyAllWindows()
