import face_recognition
import cv2
import os
from PIL import Image
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
known_dir = os.path.join(base_dir, "image_posetrain")

known_face_encodings = []
known_face_names = []

# โหลดใบหน้าที่รู้จัก
for filename in os.listdir(known_dir):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(known_dir, filename)
        try:
            pil_img = Image.open(path).convert("RGB")
            image = np.array(pil_img)

            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Loaded: {filename}")
            else:
                print(f"No face found in: {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

# เปิดกล้อง
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    print(f"Camera frame shape: {frame.shape}, dtype: {frame.dtype}")

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # ตรวจสอบขนาดช่องสี
    if small_frame.ndim != 3 or small_frame.shape[2] != 3:
        print("Camera frame is not 3-channel color image.")
        continue

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
