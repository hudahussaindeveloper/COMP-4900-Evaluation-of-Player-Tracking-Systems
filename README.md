# Player Tracking System Comparison

A Google Colab notebook for comparing and evaluating three different player tracking systems for football/soccer video analysis. This project provides a comprehensive framework for benchmarking detection accuracy, tracking continuity, and inter-system agreement across multiple tracking pipelines.

## Overview

This notebook compares three player tracking systems:

| System | Detection Model | Tracker | Weights |
|--------|-----------------|---------|---------|
| **Darkmyter** | YOLOv8-L | ByteTrack (ifzhang/ByteTrack) | Football-specific fine-tuned |
| **Eagle** | YOLOv8 | BoT-SORT + HRNet (field keypoints) | Standard COCO |
| **YOLO11 + BoT-SORT** | YOLO11-M | BoT-SORT (Ultralytics) | Standard COCO |

## Features

- Automated repository cloning and dependency installation
- Video download from Google Drive
- Flexible video selection and clip extraction (start, middle, end segments)
- Parallel evaluation across multiple tracking systems
- Comprehensive quantitative metrics
- Inter-system agreement analysis
- Visual comparison tools (overlay videos, side-by-side frames)
- Report generation with publication-ready figures
- Progress saving and resume capability
- Google Drive integration for result storage

## Quick Start

1. Open the notebook in Google Colab (GPU runtime recommended)
2. Run cells sequentially from top to bottom
3. Upload videos or use the provided Google Drive folder
4. Select videos and configure evaluation options
5. Run tracking systems and view results

## Notebook Structure

The notebook is organized into 17 cells covering the complete evaluation pipeline:

### Setup Phase (Cells 1-5)

**Cell 1: Directory Setup**
- Creates working directory structure at `/content/`
- Directories: `repositories/`, `videos/`, `clips/`, `output/`
- Defines colored status message utilities

**Cell 2: Repository Cloning**
- Clones Eagle repository from `nreHieW/Eagle`
- Clones Darkmyter repository from `Darkmyter/Football-Players-Tracking`

**Cell 3: Dependency Installation**
- Core: PyTorch, torchvision, tracklab
- Computer vision: OpenCV, NumPy, SciPy, scikit-learn
- Detection: Ultralytics, supervision
- Tracking: cython_bbox, lap, onemetric, YOLOX
- Utilities: gdown, Pillow, tqdm, psutil

**Cell 4: YOLO Weights Download**
- Downloads YOLO11-M weights (`yolo11m.pt`)
- Ensures model availability before processing

**Cell 5: Video Download**
- Downloads videos from a shared Google Drive folder
- Lists available videos with file sizes
- Supports common formats: MP4, AVI, MOV, MKV

### Configuration Phase (Cell 6)

**Cell 6: Video Selection and Clip Extraction**
- Interactive video selection (single, multiple, or all)
- Optional clip extraction: 60-second segments from start, middle, and end
- Automatic handling of videos shorter than clip duration
- FFmpeg-based clip extraction with copy codec (fast, lossless)

### System Setup Phase (Cells 7-10)

**Cell 7: Darkmyter Setup**
- Clones original ByteTrack repository (ifzhang/ByteTrack)
- Downloads football-specific YOLOv8-L weights from Roboflow football-players-detection dataset
- Creates wrapper script with ByteTrack parameters:
  - `track_thresh`: 0.25
  - `track_buffer`: 30
  - `match_thresh`: 0.8
  - `aspect_ratio_thresh`: 3.0
  - `min_box_area`: 1.0

**Cell 8: ByteTrack NumPy Patch**
- Patches deprecated NumPy type aliases (`np.float`, `np.int`)
- Ensures compatibility with NumPy 1.24+

**Cell 9: YOLO11 + BoT-SORT Setup**
- Creates runner script for Ultralytics YOLO tracking
- Configures tuned BoT-SORT parameters (sourced from [TrackLab](https://github.com/TrackingLaboratory/tracklab)):
  - `track_high_thresh`: 0.338
  - `new_track_thresh`: 0.211
  - `track_buffer`: 60
  - `match_thresh`: 0.227
  - `proximity_thresh`: 0.595
  - `appearance_thresh`: 0.482
  - `with_reid`: True

**Cell 10: Eagle Setup**
- Creates wrapper for Eagle tracking system
- Handles Eagle's hierarchical JSON output format
- Integrates HRNet for field keypoint detection, enabling automatic homography calculation

### Evaluation Phase (Cell 11)

**Cell 11: Run Tracking Evaluation**
- Executes all tracking systems on selected videos
- Monitors RAM and GPU memory usage
- Implements progress saving after each system completion
- Supports resume from interrupted evaluations
- Tracks per-system statistics: success rate, timing, detection counts

### Results Phase (Cells 12-13)

**Cell 12: Download Results**
- Creates ZIP archive of all outputs
- Downloads results via Colab file browser

**Cell 13: Load Pre-computed Results**
- Downloads cached results from Google Drive
- Useful for skipping lengthy evaluation runs

### Analysis Phase (Cells 14-15)

**Cell 14: Shared Parsers and Data Structures**

Defines the unified evaluation framework:

**Detection Format** (normalized across all systems):
```python
{
    "frame_id": int,      # 0-indexed frame number
    "track_id": int,      # Unique identifier
    "bbox": [x1, y1, x2, y2],  # Pixel coordinates
    "score": float,       # Confidence (0.0-1.0)
    "class_name": str     # "player", "referee", "ball", "goalkeeper"
}
```

**QuantitativeMetrics Class** - Per-system evaluation:
- **Trajectory Smoothness (Jerk)**: Third derivative of position; lower = smoother tracks
- **Speed Plausibility**: Violation rate for speeds >40 km/h and accelerations >5 m/s^2
- **Detection Completeness**: Players per frame, coverage rate (frames with 20-22 players)
- **Track Continuity**: Mean track length, fragmentation rate

**InterSystemAgreement Class** - Pairwise system comparison:
- **Position Distance**: Euclidean distance between matched detection centers
- **Bounding Box IoU**: Intersection over Union for matched boxes
- **Velocity Correlation**: Frame-to-frame displacement comparison
- **Disagreement Zones**: Frames with significant detection count differences

**Cell 15: Report Generation**
- Generates publication-ready visualizations using matplotlib/seaborn
- Creates figures:
  1. Executive Summary (6-panel overview)
  2. Inter-System Agreement Analysis
  3. Radar Comparison Chart
  4. Detection Statistics
  5. Agreement Heatmap
  6. Speed Analysis
  7. Trade-off Analysis (detection vs tracking)
- Exports all figures as high-DPI PNG files

### Visualization Phase (Cell 16)

**Cell 16: Video Visualization Module**

Interactive visualization tools:

**Quick Functions**:
```python
quick_all('VIDEO_NAME')        # Overlay + Solo, full video
quick_all('VIDEO_NAME', 5.0)   # Overlay + Solo, 5 minutes
quick_overlay('VIDEO_NAME')    # Overlay only
quick_solo('VIDEO_NAME')       # Solo videos only
quick_frame('VIDEO_NAME', 100) # Extract frame 100
```

**Features**:
- Color-coded bounding boxes per system (Pink=Darkmyter, Gold=Eagle, Blue=YOLO11)
- Track ID labels with system prefix
- Frame counter overlay
- Legend display
- Google Drive auto-save option

### Export Phase (Cell 17)

**Cell 17: Download Outputs**
- Creates final ZIP archive of comparison outputs
- Handles empty directory edge cases

## Output Files

### Tracking Results

Each system produces a JSON file per video/clip:
```
output/
  {video_name}/
    {clip}_eagle.json
    {clip}_darkmyter.json
    {clip}_yolo11_botsort.json
```

### Evaluation Metrics

**Aggregated CSVs**:
- `aggregated_quantitative_metrics.csv`: Per-system metrics
- `aggregated_agreement_metrics.csv`: Pairwise agreement metrics

**Full Results**:
- `evaluation_results.json`: Complete evaluation data
- `multi_clip_full_results.json`: Per-clip detailed results

### Visualizations

**Report Figures** (`comparison_output/report/`):
- `01_executive_summary.png`
- `02_inter_system_agreement.png`
- `03_radar_comparison.png`
- `04_detection_statistics.png`
- `05_agreement_heatmap.png`
- `06_speed_analysis.png`
- `07_tradeoff_analysis.png`

**Video Outputs** (`visualization_output/`):
- `{video}_overlay.mp4`: All systems overlaid
- `{video}_{system}.mp4`: Individual system videos
- `frame_{N}.png`: Extracted comparison frames

## Metrics Reference

### Quantitative Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `jerk_mean` | Mean trajectory jerk (smoothness) | Lower is better |
| `speed_mean_ms` | Average speed in m/s | 2-4 m/s typical |
| `speed_violation_rate` | Fraction >40 km/h | <5% |
| `accel_violation_rate` | Fraction >5 m/s^2 | <50% |
| `mean_players_per_frame` | Average detections | ~20-22 |
| `coverage_rate` | Frames with 20-22 players | Higher is better |
| `mean_track_length` | Average track duration (frames) | Higher is better |
| `fragmentation_rate` | Short tracks (<clip duration) | Lower is better |

### Agreement Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `position_mean_px` | Mean center distance | <5 pixels |
| `iou_mean` | Mean bounding box IoU | >0.7 |
| `match_rate_sys1/2` | Fraction of detections matched | >0.8 |
| `disagreement_rate` | Frames with count diff >3 | <20% |

## Configuration Parameters

### Evaluation Parameters

```python
fps = 30.0              # Video frame rate (Darkmyter, YOLO11); Eagle uses 24 fps
pixels_per_meter = 10.0 # Scale factor for speed calculations
max_speed_ms = 11.11    # 40 km/h threshold
max_accel_ms2 = 5.0     # Acceleration threshold
iou_threshold = 0.3     # Minimum IoU for matching
```

### Tracker-Specific Parameters

**Darkmyter (ByteTrack)**:
- `track_thresh`: 0.25
- `track_buffer`: 30 frames
- `match_thresh`: 0.8

**YOLO11 (BoT-SORT)** - tuned parameters from [TrackLab](https://github.com/TrackingLaboratory/tracklab):
- `track_high_thresh`: 0.338
- `track_buffer`: 60 frames
- `match_thresh`: 0.227
- `with_reid`: True

## Requirements

### Hardware
- Google Colab with GPU runtime (T4/V100 recommended)
- Minimum 12GB GPU memory for concurrent processing
- ~20GB disk space for videos and results

### Software
- Python 3.10+
- PyTorch 2.0+
- Ultralytics 8.0+
- OpenCV 4.8+

## Typical Results Summary

Based on evaluation across 15 video clips (~85 minutes of footage):

| System | Players/Frame | Track Length | Fragmentation | Jerk Score |
|--------|---------------|--------------|---------------|------------|
| Darkmyter | 20.9 | 287 frames | 91.6% | 37,189 |
| Eagle | 19.9 | 1,562 frames | 57.6% | 36,120 |
| YOLO11+BoT-SORT | 17.7 | 22 frames | 99.9% | 15,510 |

**Key Findings**:
- Darkmyter achieves best detection completeness (~21 players/frame)
- Eagle provides best track continuity (longest tracks, lowest fragmentation)
- YOLO11 has smoothest trajectories but highest fragmentation

## Troubleshooting

**ByteTrack Import Error**:
- Ensure Cell 8 runs successfully to patch NumPy compatibility

**Out of Memory**:
- Reduce `max_minutes` parameter for clip duration
- Process videos sequentially rather than in batch

**Missing Weights**:
- Re-run Cell 4 (YOLO) and Cell 7 (Darkmyter weights)
- Check Google Drive quota for weight downloads

**Video Codec Issues**:
- Convert videos to H.264 MP4 before uploading
- Use FFmpeg: `ffmpeg -i input.mov -c:v libx264 output.mp4`

## Citation

If you use this evaluation framework, please cite the original tracking systems:

```bibtex
@misc{Eagle2024,
  author = {nreHieW},
  title = {Eagle: Football Player Tracking},
  year = {2024},
  url = {https://github.com/nreHieW/Eagle}
}

@misc{Darkmyter2023,
  author = {Darkmyter},
  title = {Football Players Tracking},
  year = {2023},
  url = {https://github.com/Darkmyter/Football-Players-Tracking}
}

@article{ByteTrack2022,
  title = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author = {Zhang, Yifu and others},
  journal = {ECCV},
  year = {2022}
}

@misc{Ultralytics2024,
  author = {Ultralytics},
  title = {YOLO11},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}

@misc{TrackLab2024,
  author = {TrackingLaboratory},
  title = {TrackLab: A Modular Framework for Multi-Object Tracking},
  year = {2024},
  url = {https://github.com/TrackingLaboratory/tracklab}
}
```

## License

This evaluation framework is provided for research purposes. Individual tracking systems retain their original licenses. Check each repository for specific terms.