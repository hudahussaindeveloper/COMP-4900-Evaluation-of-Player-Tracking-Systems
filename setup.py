"""
Quick Setup
Run this first to check everything is installed
"""

import os
from pathlib import Path
import subprocess
import sys

def setup_directories():
    print("Setting up directories...")
    
    base_dir = Path('./required')
    dirs = [
        base_dir,
        base_dir / 'systems',
        base_dir / 'videos',
        base_dir / 'results',
        base_dir / 'test_clips',
        base_dir / 'scripts'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")
    
    print("Directories created successfully")

def setup_tracking_systems():
    systems_dir = Path(__file__).parent / "required" / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    
    systems = {
        'eagle': 'https://github.com/nreHieW/Eagle.git',
        'darkmyter': 'https://github.com/Darkmyter/Football-Players-Tracking.git',
        'anshchoudhary': 'https://github.com/AnshChoudhary/Football-Tracking.git',
        'tracklab': 'https://github.com/TrackingLaboratory/tracklab.git'
    }
    
    print("Setting up tracking systems...")
    for name, repo in systems.items():
        system_path = systems_dir / name
        if not system_path.exists():
            print(f"Cloning {name}...")
            try:
                subprocess.run(['git', 'clone', repo, str(system_path)], check=True)
                print(f"Success: {name} cloned")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone {name}: {e}")
        else:
            print(f"Already exists: {name}")
    return systems_dir

def install_uv():
    print("Checking for uv package manager...")
    
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"UV already installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("Installing uv via pip...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"UV installed successfully: {result.stdout.strip()}")
            return True
        else:
            print("UV installation completed but command not found in PATH")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Failed to install uv via pip: {e}")
        return False

def download_eagle_weights(system_path):
    print("Downloading Eagle model weights...")
    
    models_dir = system_path / "eagle" / "models"
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return False
    
    weight_files = ['best.pt', 'yolov8x.pt']
    all_exist = all((models_dir / f).exists() for f in weight_files)
    
    if all_exist:
        print("Eagle weights already downloaded")
        return True
    
    try:
        import gdown
    except ImportError:
        print("gdown not installed. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown'], check=True)
        import gdown
    
    try:
        original_dir = Path.cwd()
        os.chdir(models_dir)
        
        print("Downloading weights from Google Drive...")
        
        weights_script = models_dir / "get_weights.sh"
        if weights_script.exists():
            print("Reading get_weights.sh...")
            with open(weights_script, 'r') as f:
                script_content = f.read()
            
            import re
            gdown_match = re.search(r'gdown[^\n]*?([a-zA-Z0-9_-]{25,})', script_content)
            if gdown_match:
                file_id = gdown_match.group(1).strip()
                print(f"Found file ID: {file_id}")
                
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, 'weights.zip', quiet=False)
                
                import zipfile
                print("Extracting weights...")
                with zipfile.ZipFile('weights.zip', 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                print("Eagle weights downloaded and extracted successfully")
                os.chdir(original_dir)
                return True
            else:
                print("Could not find Google Drive file ID in get_weights.sh")
        
        print("Could not automatically download weights")
        print(f"Please manually download weights to: {models_dir}")
        os.chdir(original_dir)
        return False
        
    except Exception as e:
        print(f"Error downloading Eagle weights: {e}")
        print(f"Please manually download weights to: {models_dir}")
        os.chdir(original_dir)
        return False

def fix_eagle_pyproject(system_path):
    pyproject_path = system_path / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("Eagle pyproject.toml not found")
        return False
    
    print("Fixing Eagle pyproject.toml for compatibility...")
    
    try:
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix Python version requirement
        if '>=3.13' in content:
            content = content.replace('>=3.13', '>=3.8')
            print("Fixed Python version requirement")
        
        # Add missing dependencies
        if 'pandas' not in content:
            # Insert pandas in dependencies list
            import re
            content = re.sub(
                r'(dependencies = \[[^\]]*)',
                r'\1\n    "pandas>=1.5.0",',
                content
            )
            print("Added pandas dependency")
        
        # Fix packaging structure by adding package discovery
        if '[tool.setuptools]' not in content:
            content += '\n\n[tool.setuptools]\npackages = ["eagle"]\n'
            print("Added package discovery configuration")
        
        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("Eagle pyproject.toml fixed successfully")
        return True
        
    except Exception as e:
        print(f"Failed to fix Eagle pyproject.toml: {e}")
        return False

def create_eagle_setup_py(system_path):
    setup_py_content = '''from setuptools import setup, find_packages

setup(
    name="eagle",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "ultralytics>=8.0.0",
        "supervision>=0.1.0",
        "albumentations>=1.0.0",
        "gdown>=4.0.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
    ],
    python_requires=">=3.8",
)
'''
    
    setup_path = system_path / "setup.py"
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(setup_py_content)
    
    print("Created setup.py for Eagle")
    return setup_path

def install_eagle_dependencies_manual(system_path):
    print("Installing Eagle dependencies manually...")
    
    original_dir = Path.cwd()
    os.chdir(system_path)
    
    try:
        # Create virtual environment
        venv_path = system_path / '.venv'
        if not venv_path.exists():
            print("Creating virtual environment for Eagle...")
            subprocess.run(['uv', 'venv', '--python', sys.executable], 
                         capture_output=True, text=True)
        
        # Install core dependencies
        print("Installing core dependencies...")
        dependencies = [
            'torch', 'torchvision', 'opencv-python', 'numpy', 'scipy',
            'pandas', 'scikit-learn', 'ultralytics', 'supervision',
            'albumentations', 'gdown', 'Pillow', 'tqdm', 'requests'
        ]
        
        result = subprocess.run(['uv', 'pip', 'install'] + dependencies,
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Dependency installation failed: {result.stderr}")
            return False
        
        print("Core dependencies installed successfully")
        
        # Test if Eagle can be imported
        print("Testing Eagle imports...")
        test_script = """
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from eagle.models import CoordinateModel
    from eagle.processor import Processor
    from eagle.utils.io import read_video
    print("SUCCESS: All Eagle imports work")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"OTHER ERROR: {e}")
    sys.exit(1)
"""
        test_result = subprocess.run([
            'uv', 'run', 'python', '-c', test_script
        ], capture_output=True, text=True)
        
        if test_result.returncode == 0:
            print("Eagle installation verified")
            return True
        else:
            print(f"Eagle import test failed: {test_result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error during Eagle installation: {e}")
        return False
    finally:
        os.chdir(original_dir)

def write_eagle_wrapper(system_path):
    wrapper_code = '''#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Eagle Tracking Wrapper')
    parser.add_argument('--video_path', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()
    
    print(f"Eagle Wrapper: Processing {args.video_path}")
    
    try:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        sys.path.insert(0, str(Path(__file__).parent))
        
        from eagle.models import CoordinateModel
        from eagle.processor import Processor
        from eagle.utils.io import read_video
        import pandas as pd

        fps = 24
        frames, fps = read_video(args.video_path, fps)
        model = CoordinateModel()
        coordinates = model.get_coordinates(frames, fps, num_homography=1, num_keypoint_detection=3)

        print("Eagle: Processing data...")
        processor = Processor(coordinates, frames, fps, filter_ball_detections=False)
        df, team_mapping = processor.process_data(smooth=False)

        output = {}
        cols = [x for x in df.columns if "video" in x and x not in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]]

        for i, row in df.iterrows():
            frame_results = []
            for col in cols:
                if pd.isna(row[col]):
                    continue
                x, y = row[col]
                track_id = int(col.split("_")[1])
                frame_results.append({
                    "id": track_id,
                    "x": float(x),
                    "y": float(y),
                    "w": 30.0,
                    "h": 60.0
                })
            output[str(i)] = frame_results

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Eagle Wrapper: Completed! Processed {len(output)} frames")
        print(f"Output saved to: {output_path}")

    except ImportError as e:
        print(f"Eagle import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Eagle processing error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

    wrapper_path = system_path / "eagle_wrapper.py"
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print("Created Eagle wrapper: eagle_wrapper.py")

def write_tracklab_wrapper(system_path):
    wrapper_code = """#!/usr/bin/env python3
import argparse
import json
import sys

try:
    from tracklab import Tracker
except ImportError:
    print('Error: tracklab not installed. Run setup.py first.', file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='TrackLab wrapper for evaluation')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    print(f'Running TrackLab on: {args.video}')
    print(f'Output will be saved to: {args.output}')

    try:
        tracker = Tracker(detector='yolov8', tracker='bytetrack')
        results = tracker.run(args.video)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print('TrackLab completed successfully')
    except Exception as e:
        print(f'TrackLab error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

    wrapper_path = system_path / "run_tracklab.py"
    with open(wrapper_path, "w", encoding='utf-8') as f:
        f.write(wrapper_code)

    print("Created TrackLab wrapper: run_tracklab.py")

def install_requirements(system_name, system_path):
    print(f"Installing dependencies for: {system_name.upper()}")
    
    original_dir = Path.cwd()
    
    try:
        os.chdir(system_path)

        if system_name == 'eagle':
            # Fix pyproject.toml first
            if not fix_eagle_pyproject(system_path):
                print("Failed to fix Eagle pyproject.toml")
            
            # Create setup.py as backup
            create_eagle_setup_py(system_path)
            
            # Install dependencies manually
            if install_eagle_dependencies_manual(system_path):
                print("Eagle environment set up successfully")
                download_eagle_weights(system_path)
                write_eagle_wrapper(system_path)
            else:
                print("Failed to set up Eagle dependencies")

        elif system_name == 'darkmyter':
            print("Installing Darkmyter dependencies...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'ultralytics', 'opencv-python', 'numpy'], check=False)
            print("Darkmyter dependencies installed")

        elif system_name == 'anshchoudhary':
            print("Installing AnshChoudhary dependencies...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'opencv-python', 'numpy', 'ultralytics'], check=False)
            print("AnshChoudhary dependencies installed")

        elif system_name == 'tracklab':
            if not install_uv():
                print("Cannot install TrackLab without uv")
                return
                
            print("Setting up TrackLab with uv...")
            
            venv_path = system_path / '.venv'
            if not venv_path.exists():
                print("Creating virtual environment for TrackLab...")
                subprocess.run(['uv', 'venv', '--python', sys.executable], 
                             cwd=system_path, check=False)
            
            subprocess.run(['uv', 'pip', 'install', '-e', '.'], 
                         cwd=system_path, check=False)
            
            print("TrackLab environment set up successfully")
            write_tracklab_wrapper(system_path)

    except Exception as e:
        print(f"Error installing {system_name}: {e}")
    finally:
        os.chdir(original_dir)

def main():
    base_dir = Path(__file__).parent
    systems_dir = base_dir / 'required' / 'systems'
    
    print("PLAYER TRACKING SYSTEMS SETUP")
    
    install_uv()
    setup_directories()
    setup_tracking_systems()
    systems = ['eagle', 'darkmyter', 'anshchoudhary', 'tracklab']
    
    for system in systems:
        system_path = systems_dir / system
        if system_path.exists():
            install_requirements(system, system_path)
        else:
            print(f"System not found: {system} at {system_path}")
    
    print("SETUP COMPLETE!")
    print("Next steps:")
    print("1. Place your video files in: required/videos/")
    print("2. Update tracking_system_evaluator.py to use eagle_wrapper.py")
    print("3. Run the evaluator: python tracking_system_evaluator.py")

if __name__ == '__main__':
    main()