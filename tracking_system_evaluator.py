#!/usr/bin/env python3
"""
Simple Player Tracking System Evaluator
Tests and compares 4 tracking systems on football videos
"""

import os
import json
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import cv2

# Root directory of the script
BASE_DIR = Path(__file__).resolve().parent

# Config: relative paths and commands for each system
SYSTEMS_CONFIG = {
    'eagle': {
        'path': BASE_DIR / 'required' / 'systems' / 'eagle',
        'command': 'python eagle_wrapper.py --video_path {video} --output {output}',
        'use_wrapper': True
    },
    'darkmyter': {
        'path': BASE_DIR / 'required' / 'systems' / 'darkmyter',
        'command': 'python inference.py --source {video} --output {output}'
    },
    'anshchoudhary': {
        'path': BASE_DIR / 'required' / 'systems' / 'anshchoudhary',
        'command': 'python yolo_inference.py --source {video} --output {output}'
    },
    'tracklab': {
        'path': BASE_DIR / 'required' / 'systems' / 'tracklab',
        'command': 'uv run --no-sync run_tracklab.py --video {video} --output {output}'
    }
}

def run_tracking_system(system_name: str, video_path: Path, output_dir: Path) -> Dict:
    config = SYSTEMS_CONFIG.get(system_name)
    if not config:
        return {'tracks': {}, 'time': 0, 'error': f'System {system_name} not configured'}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{system_name}_output.json'

    # Build command
    cmd = config['command'].format(
        video=f'"{str(video_path)}"',
        output=f'"{str(output_file)}"'
    )

    start_time = time.time()
    try:
        original_dir = Path.cwd()
        os.chdir(config['path'])

        print(f"→ Running: {cmd}")
        print(f"→ In: {config['path'].as_posix()}")
        print(f"→ Output expected at: {output_file.as_posix()}")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)

        os.chdir(original_dir)

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return {'tracks': {}, 'time': time.time() - start_time,
                    'error': f'Command failed: {error_msg}'}

        tracks = parse_tracking_output(output_file, system_name)
        return {'tracks': tracks, 'time': time.time() - start_time, 'error': None}

    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        return {'tracks': {}, 'time': 600, 'error': 'Timeout (10 minutes)'}
    except Exception as e:
        os.chdir(original_dir)
        return {'tracks': {}, 'time': time.time() - start_time, 'error': str(e)}


def parse_tracking_output(output_file: Path, system_name: str) -> Dict:
    """Parse tracking output and return dictionary of {frame_id: [player_detections]}"""
    tracks = {}

    if not output_file.exists():
        print(f"✗ Output file not found for {system_name}: {output_file.as_posix()}")
        return tracks

    try:
        if system_name in ['eagle', 'anshchoudhary', 'tracklab']:
            with output_file.open('r') as f:
                data = json.load(f)
                
                # Eagle format: {frame_id: [player_data]}
                if isinstance(data, dict):
                    for frame_id, players in data.items():
                        if isinstance(players, list):
                            tracks[int(frame_id)] = players
                
                print(f"✓ Parsed {len(tracks)} frames from {system_name}")

        elif system_name == 'darkmyter':
            with output_file.open('r') as f:
                data = json.load(f)
                # Assuming similar format
                if isinstance(data, dict):
                    for frame_id, players in data.items():
                        if isinstance(players, list):
                            tracks[int(frame_id)] = players
                            
                print(f"✓ Parsed {len(tracks)} frames from {system_name}")

    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error for {system_name}: {e}")
    except Exception as e:
        print(f"✗ Could not parse output for {system_name}: {e}")

    return tracks


def evaluate_video(video_path: Path, systems: List[str] = None, output_dir: Path = BASE_DIR / 'results') -> Dict:
    if not systems:
        systems = list(SYSTEMS_CONFIG.keys())

    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {video_path.name}")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for system in systems:
        print(f"\n{system.upper()}:")
        system_output = output_dir / system
        result = run_tracking_system(system, video_path, system_output)
        results[system] = result

        if result['error']:
            print(f"  ✗ Error: {result['error']}")
            result['metrics'] = {
                'detection_rate': 0,
                'avg_players': 0,
                'track_smoothness': float('inf'),
                'total_frames': 0
            }
        else:
            print(f"✓ Completed in {result['time']:.1f}s")
            result['metrics'] = calculate_metrics(result['tracks'])

    save_results(results, output_dir)
    return results


def calculate_metrics(tracks: Dict, fps: float = 25.0) -> Dict:
    if not tracks:
        return {
            'detection_rate': 0,
            'avg_players': 0,
            'track_smoothness': float('inf'),
            'total_frames': 0
        }

    good_frames = 0
    total_players = 0
    for frame_id, players in tracks.items():
        num_players = len(players)
        total_players += num_players
        if 20 <= num_players <= 22:
            good_frames += 1

    detection_rate = (good_frames / len(tracks)) * 100 if len(tracks) > 0 else 0
    avg_players = total_players / len(tracks) if len(tracks) > 0 else 0

    # Smoothness (placeholder)
    track_smoothness = 0.0

    return {
        'detection_rate': detection_rate,
        'avg_players': avg_players,
        'track_smoothness': track_smoothness,
        'total_frames': len(tracks)
    }


def save_results(results: Dict, output_dir: Path):
    comparison = []

    for system, result in results.items():
        m = result['metrics']
        time_s = result['time']
        fps = m['total_frames'] / time_s if time_s > 0 else 0
        comparison.append({
            'System': system,
            'Detection Rate (%)': f"{m['detection_rate']:.1f}",
            'Avg Players': f"{m['avg_players']:.1f}",
            'Smoothness': f"{m['track_smoothness']:.2f}",
            'Frames': m['total_frames'],
            'Time (s)': f"{time_s:.1f}",
            'FPS': f"{fps:.1f}",
            'Status': 'Valid' if not result['error'] else 'Invalid'
        })

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'results.json'
    with results_file.open('w') as f:
        json.dump({k: {
            'time': v['time'],
            'error': v['error'],
            'metrics': v['metrics']
        } for k, v in results.items()}, f, indent=2)

    print(f"\n✓ Results saved to: {results_file.as_posix()}")


def extract_clip(video_path: Path, start_time: int, duration: int, output_path: Path):
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',
        str(output_path),
        '-y'
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"✓ Extracted clip: {output_path.as_posix()}")


def test_on_clips(video_path: Path, clip_duration: int = 120):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    total_duration = total_frames / fps

    print(f"\nVideo info:")
    print(f"  Duration: {total_duration:.1f} seconds")
    print(f"  FPS: {fps:.1f}")

    if total_duration < clip_duration * 3:
        print("✗ Video is too short for 3 clips. Using full video instead.")
        evaluate_video(video_path, output_dir=BASE_DIR / 'results' / 'full_video')
        return

    start_times = [
        0,
        max((total_duration - clip_duration) / 2, 0),
        max(total_duration - clip_duration, 0)
    ]

    clips_dir = BASE_DIR / 'test_clips'
    clips_dir.mkdir(exist_ok=True)

    for i, start_time in enumerate(start_times):
        clip_path = clips_dir / f'clip_{i+1}.mp4'
        print(f"\nExtracting clip {i+1}/3 (start: {start_time:.1f}s)...")
        extract_clip(video_path, start_time, clip_duration, clip_path)
        print(f"\nEvaluating clip {i+1}...")
        evaluate_video(clip_path, output_dir=BASE_DIR / 'results' / f'clip_{i+1}')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Player tracking system evaluator')
    parser.add_argument('video', nargs='?', help='Path to a specific video file (optional)')
    parser.add_argument('--systems', nargs='+', choices=list(SYSTEMS_CONFIG.keys()),
                        help='Systems to test (default: all)')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--clips', action='store_true',
                        help='Run on 3 clips (start, middle, end) instead of full video')
    parser.add_argument('--duration', type=int, default=120,
                        help='Clip duration in seconds (default: 120)')

    args = parser.parse_args()

    if args.video:
        video_files = [Path(args.video)]
    else:
        videos_path = BASE_DIR / 'required' / 'videos'
        video_files = sorted(videos_path.glob('*.mp4'))
        if not video_files:
            print("✗ No videos found in required/videos/")
            return

    for video_path in video_files:
        name = video_path.stem
        print(f"\n\n🎬 Processing video: {name}")
        if args.clips:
            test_on_clips(video_path, args.duration)
        else:
            evaluate_video(video_path, args.systems, output_dir=BASE_DIR / args.output / name)


if __name__ == '__main__':
    main()
