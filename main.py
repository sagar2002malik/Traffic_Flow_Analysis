import sys
import os
import json
import cv2
import numpy as np
import yt_dlp
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def download_video(url, output_path="video/video.mp4"):
    """Download YouTube video into video/ folder and return its path"""
    try:
        os.makedirs("video", exist_ok=True)
        ydl_opts = {
            "format": "bestvideo[height=720]/bestvideo[height=1080]",
            "outtmpl": output_path,
            "nopart": True,
            "continuedl": False,
            "retries": 3
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ Video downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        sys.exit(1)

def run_detection(video_path):
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model
    model = YOLO("models/yolo11x.pt").to(device)
    if device == "cuda":
        model.fuse()
        model.half()

    # Load lane polygons
    with open("lanes/lanes.json", "r") as f:
        lanes = json.load(f)

    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)
    lane_counts = {lane["name"]: set() for lane in lanes}

    os.makedirs("output", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join("output", "output_with_deepsort.mp4")
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with torch.inference_mode():
            results = model.predict(frame, conf=0.5, verbose=False, device=device, half=(device == "cuda"))

        detections = []
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy
            confs = boxes.conf
            classes = boxes.cls.int()

            for i in range(len(xyxy)):
                cls = int(classes[i].item())
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    conf = float(confs[i].item())
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    inside_lane = False
                    for lane in lanes:
                        points = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in lane["polygon"]]
                        if cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False) >= 0:
                            inside_lane = True
                            break

                    if inside_lane:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            for lane in lanes:
                points = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in lane["polygon"]]
                if cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False) >= 0:
                    lane_counts[lane["name"]].add(track_id)

        for lane in lanes:
            points = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in lane["polygon"]]
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

            M = cv2.moments(np.array(points, np.int32))
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cx, cy = points[0]

            cv2.putText(frame, f"{lane['name']} | Count: {len(lane_counts[lane['name']])}",
                        (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Vehicle Tracking with DeepSORT & Lane Counts", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    txt_path = os.path.join("output", "lane_counts.txt")
    with open(txt_path, "w") as f:
        for lane, ids in lane_counts.items():
            f.write(f"{lane}: {len(ids)}\n")

    print(f"✅ Video saved to {out_path}")
    print(f"✅ Lane counts saved to {txt_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <YouTube_URL>")
        sys.exit(1)

    url = sys.argv[1]

    # Step 1: Download video
    video_path = download_video(url)

    # Step 2: Run detection
    run_detection(video_path)

if __name__ == "__main__":
    main()
