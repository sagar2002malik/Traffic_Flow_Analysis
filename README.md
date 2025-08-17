# Traffic_Flow_Analysis
🚗 Vehicle detection and lane-based counting using YOLOv11 and DeepSORT. Tracks cars, bikes, buses, and trucks in video, assigns unique IDs, and counts vehicles per lane with real-time visualization.

This project downloads a video from **YouTube**, detects and tracks
vehicles, assigns unique IDs using **DeepSORT**, and counts vehicles
inside user-defined lanes. It uses **YOLOv11** for object detection and
**OpenCV** for video processing.

------------------------------------------------------------------------

## 📂 Project Structure

    project-root/
    │── video/                     # Downloaded video will be saved here
    │   └── video.mp4
    │
    │── lanes/
    │   └── lanes.json             # Lane polygons (normalized coordinates)
    │
    │── models/
    │   ├── yolo11s.pt             # YOLOv11 small model
    │   ├── yolo11x.pt             # YOLOv11 extra-large model (default)
    │   ├── yolov8l.pt             # YOLOv8 large model
    │   ├── yolov8m.pt             # YOLOv8 medium model
    │   └── yolov8n.pt             # YOLOv8 nano model
    │
    │── output/
    │   ├── lane_counts.txt        # Final lane-wise vehicle counts
    │   └── output_with_deepsort.mp4  # Processed output video
    │
    │── venv/                      # Virtual environment (optional)
    │── main.py                    # Main script (download + detection)
    │── requirements.txt           # Python dependencies
    │── README.md                  # Project documentation

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1. Clone Repository (or Download Code)

``` bash
git clone https://github.com/sagar2002malik/Traffic_Flow_Analysis.git
cd Traffic_Flow_Analysis
```

### 2. Create Virtual Environment (Optional but Recommended)

**Windows:**

``` bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📥 Input Files

1.  **YouTube Video Input**
    -   No need to manually place a video.\
    -   Just provide a **YouTube URL** when running the script.\
    -   The video will be downloaded to `video/video.mp4`.
2.  **Lane Definitions (`lanes.json`)**
    -   Define lanes as polygons in **normalized coordinates (0--1)**.\

    -   Example:

        ``` json
        [
          {
            "name": "Lane 1",
            "polygon": [
              [0.2, 0.1],
              [0.4, 0.1],
              [0.4, 0.9],
              [0.2, 0.9]
            ]
          },
          {
            "name": "Lane 2",
            "polygon": [
              [0.5, 0.1],
              [0.7, 0.1],
              [0.7, 0.9],
              [0.5, 0.9]
            ]
          }
        ]
        ```

------------------------------------------------------------------------

## ▶️ Running the Script

Run with a YouTube URL:

``` bash
python main.py "https://www.youtube.com/watch?v=MNn9qKG2UFI"
```

### Script Features:

-   Downloads video → `video/video.mp4`\
-   Loads **YOLOv11x** model (`models/yolo11x.pt`)\
-   Runs detection on **vehicles only** (car, motorcycle, bus, truck)\
-   Tracks objects using **DeepSORT**\
-   Assigns vehicles to lanes based on polygon definitions\
-   Displays live **vehicle counts per lane**\
-   Saves results:
    -   Processed video → `output/output_with_deepsort.mp4`\
    -   Lane counts → `output/lane_counts.txt`

------------------------------------------------------------------------

## 📊 Output Example

**lane_counts.txt**

    Lane 1: 34
    Lane 2: 27
    Lane 3: 12

**Processed video:**\
- Bounding boxes with **unique track IDs**\
- Lane outlines and real-time counts\
- FPS overlay

------------------------------------------------------------------------

## 🚀 Performance Notes

-   Runs on **GPU** if available (`CUDA + FP16` inference).\
-   Falls back to **CPU** if GPU is not detected.\
-   For faster results, use a smaller YOLO model (`yolo11s.pt` or
    `yolov8n.pt`).

------------------------------------------------------------------------

## 🛠️ Requirements

Main dependencies (see `requirements.txt`):\
- `yt-dlp`\
- `ultralytics`\
- `opencv-python`\
- `numpy`\
- `torch`\
- `deep-sort-realtime`

------------------------------------------------------------------------

## 🎯 Future Improvements

-   Support multiple YouTube links in a single run\
-   Option to auto-delete downloaded video after processing\
-   Real-time webcam/live-stream support\
-   Integration with dashboards for analytics
