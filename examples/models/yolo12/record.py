import cv2

def record_webcam_to_mp4(output_filename="output.mp4", fps=30.0):
    """
    Records video from the default webcam and saves it to an MP4 file.

    Args:
        output_filename (str): The name of the output MP4 file.
        fps (float): Frames per second for the output video.
    """
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get the default frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    # 'mp4v' is a common codec for MP4 files. Other options like 'XVID' for .avi might be used.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    print(f"Recording started. Press 'q' to stop and save to {output_filename}...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame (optional)
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to exit the loop and stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped and saved.")

if __name__ == "__main__":
    record_webcam_to_mp4()
