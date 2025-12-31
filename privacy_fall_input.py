from ultralytics import YOLO
import cv2
import numpy as np
import serial

# Initialize 
try:
    ser = serial.Serial("COM6", 9600)  # Replace with your Arduino's port
    print("Serial connection established.")
except Exception as e:
    print(f"Error establishing serial connection: {e}")
    ser = None

# Load
model = YOLO('yolov8n-pose.pt')

# Dictionary to store previous keypoints and hand distances
prev_keypoints_dict = {}
prev_hand_distances = {}

# Define action recognition function
def classify_action(keypoints, prev_keypoints=None):
    if len(keypoints) < 17:
        return "Other"

    nose = keypoints[0]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hand, right_hand = keypoints[9], keypoints[10]
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_knee, right_knee = keypoints[13], keypoints[14]
    left_ankle, right_ankle = keypoints[15], keypoints[16]

    hip_height = (left_hip[1] + right_hip[1]) / 2
    knee_height = (left_knee[1] + right_knee[1]) / 2
    ankle_height = (left_ankle[1] + right_ankle[1]) / 2
    shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
    nose_height = nose[1]

    # Detect "Sleep"
    sleep_threshold = 30
    if (abs(nose_height - shoulder_height) < sleep_threshold and
        abs(shoulder_height - hip_height) < sleep_threshold):
        return "Sleep"

    # Detect "Clap"
    clap_threshold = 50
    min_movement = 5
    prev_clap_distance = prev_hand_distances.get("clap", None)
    clap_distance = np.linalg.norm(np.array(left_hand) - np.array(right_hand))

    if prev_clap_distance is not None:
        hand_motion = prev_clap_distance - clap_distance
        if prev_clap_distance > clap_threshold and clap_distance < clap_threshold and hand_motion > min_movement:
            return "Clap"

    prev_hand_distances["clap"] = clap_distance

    # Detect "Wave"
    wave_threshold = nose[1]
    if left_hand[1] < wave_threshold or right_hand[1] < wave_threshold:
        return "Wave"

    if prev_keypoints is not None:
        prev_hip_height = (prev_keypoints[11][1] + prev_keypoints[12][1]) / 2
        upward_motion = prev_hip_height - hip_height > 30
    else:
        upward_motion = False

    sitting_threshold = abs(hip_height - knee_height) < 30
    standing_threshold = abs(hip_height - shoulder_height) > 50 and abs(hip_height - ankle_height) > 100
    walk_threshold = abs(left_ankle[1] - right_ankle[1]) > 30

    if upward_motion and hip_height < shoulder_height:
        return "Jump"
    elif sitting_threshold:
        return "Sit"
    elif standing_threshold and not walk_threshold:
        return "Stand"
    elif walk_threshold:
        return "Walk"
    elif hip_height < knee_height:
        return "Fall"
    else:
        return "Other"

# Open video file instead of webcam
video_path = "Fall.mp4"  # Update with your actual video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open the video file {video_path}.")
    exit()

resize_width = 640
resize_height = 360

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))
    original_frame = frame.copy()  # Unaltered frame for "Live Video" window

    # Perform inference
    results = model(frame)

    # Create black background for "Skeleton View"
    skeleton_frame = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)

    if hasattr(results[0], "keypoints"):
        keypoints_all = results[0].keypoints.xy.cpu().numpy()

        for idx, keypoints in enumerate(keypoints_all):
            action = classify_action(keypoints, prev_keypoints_dict.get(idx))
            prev_keypoints_dict[idx] = keypoints

            if action == "Fall":
                print(f"Fall detected for Person {idx + 1}")
                if ser:
                    ser.write(b'A')

            if action == "Clap":
                print(f"Clap detected for Person {idx + 1}")

            if action == "Wave":
                print(f"Wave detected for Person {idx + 1}")

            if action == "Sleep":
                print(f"Sleep detected for Person {idx + 1}")

            # Draw keypoints on black background (Skeleton View)
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(skeleton_frame, (x, y), 5, (0, 255, 255), -1)  # Yellow keypoints

            # Draw skeleton connections
            pairs = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14),
                     (13, 15), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10)]  # Pose connections

            for p1, p2 in pairs:
                if p1 < len(keypoints) and p2 < len(keypoints):
                    x1, y1 = int(keypoints[p1][0]), int(keypoints[p1][1])
                    x2, y2 = int(keypoints[p2][0]), int(keypoints[p2][1])
                    cv2.line(skeleton_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines

            # Display action label in "Skeleton View"
            x, y = int(keypoints[0][0]), int(keypoints[0][1])  # Nose keypoint as reference
            cv2.putText(skeleton_frame, f"Person {idx + 1}: {action}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show both windows
    cv2.imshow('Live Video', original_frame)  # Unaltered video feed
    cv2.imshow('Skeleton View', skeleton_frame)  # Skeleton + Action labels

    #if cv2.waitKey(1) & 0xFF == ord('q'):
       # break

#cap.release()
#if ser:
    #ser.close()
#cv2.destroyAllWindows()