# Real-Time Hand Gesture Recognition & Industrial Work Cycle Monitoring

An embedded computer vision system for real-time hand tracking, object detection, and automated industrial work cycle monitoring. Developed using **MediaPipe, OpenCV, TensorFlow Lite, and Flask**, the system was initially implemented on Raspberry Pi 4 and subsequently migrated to NVIDIA Jetson Nano to support improved real-time processing capabilities.

## Project Overview

This project implements an embedded computer vision pipeline to monitor industrial workstation operations by automatically identifying three stages of a work cycle:

**Pick → Process → Place**

The system detects hand movements and positions (left, right, up, down) and provides real-time visualization through a Flask-based web dashboard.

Developed during an internship at **Novation City**, an Industry 4.0 excellence center in Sousse, Tunisia, as part of the Industrial Electronics Engineering curriculum at the National Engineering School of Sousse (ENISo).

## Key Features

- Real-time hand detection and tracking using **MediaPipe Hands**
- Object detection using **EfficientDet-Lite0 (TensorFlow Lite)**
- Work cycle recognition through a **finite state machine (FSM)**
- Hand position classification: left, right, up, and down
- Automated cycle logging in **JSON format**
- **Flask monitoring dashboard** featuring MJPEG video streaming, Start/Stop controls, and manual cycle validation

## Hardware Platforms

| Platform           | Implementation                                                                  |
| ------------------ | ------------------------------------------------------------------------------- |
| Raspberry Pi 4     | Complete computer vision pipeline validated                                     |
| NVIDIA Jetson Nano | Auvidea JN30D carrier board, L4T R32.7.1, JetPack 4.6.x, Ubuntu 18.04 (aarch64) |
| Camera             | USB webcam connected directly to the board                                      |

**Inference:** Currently performed on the CPU using XNNPACK. TensorRT optimization for GPU-accelerated inference on Jetson Nano is planned as a future improvement.

## Repository Structure

```text
cycle-monitoring-vision/
├── scripts/
│   ├── app.py
│   └── templates/
│       └── dashboard.html
├── models/
│   └── [EfficientDet-Lite0 TFLite model]
├── logs/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

- `scripts/`: Computer vision pipeline and Flask dashboard
- `models/`: TensorFlow Lite object detection models
- `logs/`: Generated JSON work cycle logs (excluded from version control)
- `docs/`: Technical documentation, system architecture, and diagrams

## Installation

### 1. Python Environment

Create and activate a Conda environment using Miniforge:

```bash
conda create -n gesture_env python=3.11 -y
conda activate gesture_env
pip install -r requirements.txt
```

After restarting the device:

```bash
source ~/.bashrc
conda activate gesture_env
```

### 2. NVIDIA Jetson Nano Configuration

The Jetson Nano implementation uses the **Auvidea JN30D carrier board** with the corresponding Auvidea Board Support Package (BSP).

**Software environment:**

- L4T R32.7.1 / JetPack 4.6.x
- Ubuntu 18.04 (aarch64)
- MediaPipe 0.10.18
- OpenCV 4.10.0.84
- NumPy 1.26.4

Additional environment configuration may be required for `libstdcxx-ng` and `libgomp.so.1` shared library compatibility.

The current implementation uses a USB webcam. CSI camera integration requires a GStreamer pipeline using `nvarguscamerasrc`.

### 3. Raspberry Pi 4 Configuration

Install the required system dependencies:

```bash
sudo apt update
sudo apt install libopenblas-dev
```

The application uses a USB camera for real-time video acquisition.

## Running the Application

Activate the environment and launch the application:

```bash
conda activate gesture_env
cd scripts
python app.py
```

Access the dashboard at:

```text
http://<device-ip>:5000
```

## Technical Implementation

**Hand Tracking:** MediaPipe Hands detects and tracks hand landmarks for real-time position analysis.

**Object Detection:** EfficientDet-Lite0 performs object detection using TensorFlow Lite inference.

**Cycle Recognition:** A finite state machine identifies transitions between the Pick, Process, and Place stages.

**Real-Time Monitoring:** Flask provides an MJPEG video stream, cycle visualization, manual validation controls, and JSON-based event logging.

**Performance Optimization:** NumPy memory-contiguous image arrays are used for compatibility with MediaPipe image processing. CPU inference uses XNNPACK, with future TensorRT integration planned for Jetson Nano GPU acceleration.

## Future Improvements

- TensorRT integration for GPU-accelerated inference
- Real-time performance benchmarking on Jetson Nano
- Improved industrial work cycle classification and monitoring

## Author

**Sahar Ghazel**
Industrial Electronics Engineering — National Engineering School of Sousse (ENISo), Tunisia

**Areas of Interest:** Embedded Systems | Edge AI | Computer Vision | Robotics | Industrial Automation
