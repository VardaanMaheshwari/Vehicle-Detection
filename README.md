---
title: Indian Vehicle Detection
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
---

# Indian Vehicle Detection & Tracking

A YOLO11n object detector fine-tuned to detect 6 vehicle classes — Ambulance, Bicycle, Bus, Car, Motorcycle, Truck — with multi-object tracking on video.

## Model
YOLO11n fine-tuned on a custom vehicle dataset.
- **mAP@50:** 0.799
- **mAP@50-95:** 0.624

## Features
- Image and video object detection
- Multi-object tracking (ByteTrack / BoTSORT) with persistent IDs across frames
- Adjustable confidence and IoU thresholds

## Limitations
- Detects only the 6 trained vehicle classes; other objects are ignored
- CPU inference, so video runs at reduced frame rate

## Tech Stack
Python, Ultralytics YOLO, OpenCV, Streamlit, Hugging Face Spaces