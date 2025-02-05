import cv2
import os

video_path = r"vid1.mp4"
output_dir = "frames/"

# Check if video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video '{video_path}' ne postoji")

vid = cv2.VideoCapture(video_path)

# Check if video is opened
if not vid.isOpened():
    raise ValueError(f"Video '{video_path}' se ne može otvoriti. Provjeri putanju ili format videa")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
while True:
    success, frame = vid.read()
    if not success:
        break
    if frame_count % 5 == 0:
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
    frame_count += 1

vid.release()
print("Ekstrakcija frejmova završena.")
