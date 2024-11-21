import cv2
import os

def initialize_tracker(tracker_type):
    if tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    else:
        raise ValueError('Unsupported tracker type')

def track_object(video_path, tracker_type, output_folder):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Allow user to select the initial frame to select bbox (ROI) and start tracking
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return

        # Show frame
        cv2.imshow(tracker_type, frame)
        key = cv2.waitKey(0) & 0xFF

        # If user presses 's', select ROI
        if key == ord('s'):
            bbox = cv2.selectROI(tracker_type, frame, fromCenter=False, showCrosshair=True)
            break
        # If user presses 'q', quit
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        # If user presses 'n', skip to the next frame
        elif key == ord('n'):
            continue

    # Initialize tracker
    tracker = initialize_tracker(tracker_type)
    tracker.init(frame, bbox)

    # Track object for 10 frames
    frame_count = 0
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        # Draw bounding box
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Save the frame
        output_path = os.path.join(output_folder, f"{tracker_type}_frame_{frame_count}.jpg")
        cv2.imwrite(output_path, frame)

        # Show frame
        cv2.imshow(tracker_type, frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "../data/v1.mov"
    output_folder = "output_frames"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Remove all previous saved frames from the folder
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Step 5: Track using KCF
    track_object(video_path, "KCF", output_folder)

    # Step 6: Track using CSRT
    track_object(video_path, "CSRT", output_folder)

if __name__ == "__main__":
    main()

