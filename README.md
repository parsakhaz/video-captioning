# Video Captioning with Moondream

> **⚠️ IMPORTANT:** This project currently uses Moondream 2B (2025-01-09 release) via the Hugging Face Transformers library. We will migrate to the official Moondream client libraries once they become available for this version.
>
> **⚠️ NOTE:** This project requires access to Meta's LLaMA 3.2 3B Instruct model via HuggingFace. You must request and be granted access before using this script. Visit [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) to request access.

## Table of Contents
- [Overview](#overview)
- [Sample Output](#sample-output)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Linux/macOS Installation](#linuxmacos-installation)
  - [Windows Installation](#windows-installation)
- [Usage](#usage)
  - [Command-Line Options](#command-line-options)
  - [Examples](#examples)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Dependencies](#dependencies)
- [Model Details](#model-details)
- [License](#license)

## Overview
This project generates detailed captions for videos using Moondream for frame analysis and LLaMA for scene detection and narrative generation. It intelligently extracts frames, identifies scene transitions, and creates flowing narratives with professional caption overlays.

## Sample Output
[Add GIFs or images showcasing the output once available]

## Features
- **Intelligent Frame Extraction**:
  - Time-based sampling (e.g., one frame every N seconds)
  - Total frame count (evenly spaced throughout the video)
  - Configurable extraction intervals

- **Smart Scene Detection**:
  - Word overlap analysis to detect scene transitions (33% threshold)
  - Automatic segmentation of continuous scenes
  - Frame-accurate scene boundary detection

- **Structured Caption Generation**:
  - Detailed frame analysis focusing on:
    1. People and their actions
    2. Setting and location
    3. Notable objects and activities
    4. Overall atmosphere
  - Preservation of visible text from scenes
  - Filtering of recurring elements to reduce hallucinations

- **Professional Caption Overlay**:
  - Dynamic font sizing based on video dimensions
  - Smart text wrapping with proper line breaks
  - Semi-transparent background for readability
  - Centered positioning with proper padding
  - Anti-aliased text with outline for clarity

- **Detailed Scene Analysis**:
  - JSON output with comprehensive scene information
  - Frame-by-frame caption tracking
  - Scene transition timestamps
  - Recurring elements analysis

## Prerequisites
1. Python 3.8 or later
2. CUDA-capable GPU (8GB+ VRAM recommended)
3. FFmpeg installed on your system
4. HuggingFace account with approved access to Meta's LLaMA 3.2 3B Instruct model

## Installation

### Linux/macOS Installation

1. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y libvips42 libvips-dev ffmpeg

   # CentOS/RHEL
   sudo yum install vips vips-devel ffmpeg

   # macOS
   brew install vips ffmpeg
   ```

2. Clone and setup the project:
   ```bash
   git clone [repository-url]
   cd video-captioning
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Windows Installation

Windows setup requires a few additional steps for proper GPU support and libvips installation.

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd video-captioning
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install PyTorch with CUDA support:
   ```bash
   # For NVIDIA GPUs
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install libvips: Download the appropriate version based on your system architecture:
   | Architecture | VIPS Version to Download |
   | ------------ | ------------------------ |
   | 32-bit x86   | vips-dev-w32-all-8.16.0.zip |
   | 64-bit x64   | vips-dev-w64-all-8.16.0.zip |

   - Extract the ZIP file
   - Copy all DLL files from `vips-dev-8.16\bin` to either:
     - Your project's root directory (easier) OR
     - `C:\Windows\System32` (requires admin privileges)
   - Add to PATH:
     1. Open System Properties → Advanced → Environment Variables
     2. Under System Variables, find PATH
     3. Add the full path to the `vips-dev-8.16\bin` directory

5. Install FFmpeg:
   - Download from https://ffmpeg.org/download.html#build-windows
   - Extract and add the `bin` folder to your system PATH (similar to step 4)

6. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your video files in the `inputs` folder
2. Authenticate with HuggingFace:
   ```bash
   huggingface-cli login
   ```
   Or run with a token:
   ```bash
   python video-captioning.py --token YOUR_TOKEN
   ```

3. Run the script with desired options:
   ```bash
   # Default: Extract one frame every 0.67 seconds
   python video-captioning.py

   # Custom frame interval (e.g., every 2 seconds)
   python video-captioning.py --frame-interval 2.0

   # Fixed number of total frames
   python video-captioning.py --total-frames 30
   ```

### Command-Line Options
| Option | Default | Description |
|--------|---------|-------------|
| `--token` | None | HuggingFace authentication token. If not provided, will use browser-based login or existing token. |
| `--frame-interval` | 0.67 | Time interval (in seconds) between extracted frames. Lower values mean more frames and slower processing. Recommended for most videos. |
| `--total-frames` | None | Extract a fixed number of frames evenly spaced throughout the video. Overrides `--frame-interval` if specified. Use for consistent frame count across different video lengths. |
| `--input-dir` | "./inputs" | Directory containing input videos to process. |
| `--output-dir` | "outputs" | Directory where processed videos and analysis will be saved. |

**Recommended Settings:**
- For short videos (< 2 minutes): Use default `--frame-interval 0.67` for detailed analysis
- For medium videos (2-5 minutes): Use `--frame-interval 1.0` to balance detail and processing time
- For long videos (> 5 minutes): Use `--frame-interval 2.0` or specify `--total-frames 180` for consistent processing
- For very long videos: Use `--frame-interval 5.0` or higher to avoid memory issues

### Examples
```bash
# Process video with default settings
python video-captioning.py

# Extract one frame every 5 seconds
python video-captioning.py --frame-interval 5.0

# Extract exactly 30 frames from the video
python video-captioning.py --total-frames 30

# Use custom input/output directories
python video-captioning.py --input-dir my_videos --output-dir results

# Combine multiple options
python video-captioning.py --frame-interval 2.0 --output-dir results --token YOUR_TOKEN
```

## Output
The script generates:
1. **Captioned Video**: `outputs/captioned_[original_name].mp4`
   - Original video with professional caption overlays
   - Captions change at scene transitions
   - High-quality text rendering with background

2. **Scene Analysis**: `outputs/scene_analysis.json`
   - Detailed breakdown of scenes
   - Timestamps and transitions
   - Frame-by-frame captions
   - Scene analysis with recurring elements
   - Preserved text elements from scenes

## Troubleshooting
1. CUDA/GPU Issues:
   - Ensure CUDA is properly installed
   - Verify GPU has sufficient VRAM (8GB+ recommended)
   - Close other GPU-intensive applications

2. Memory Issues:
   - Reduce frame extraction frequency
   - Process shorter video segments
   - Clear GPU cache between videos

3. Model Loading Issues:
   - Verify HuggingFace authentication
   - Check LLaMA model access status
   - Update transformers library

4. libvips Errors:
   - Make sure libvips is properly installed for your OS
   - Check system PATH includes libvips

5. Video Format Issues:
   - Ensure FFmpeg is installed and in your system PATH
   - Check video file format compatibility
   - Try converting problematic videos to MP4

## Performance Notes
- Processing time depends on:
  - Video length and resolution
  - Frame extraction interval
  - GPU memory and speed
  - Number of scene transitions

- Memory Usage:
  - Moondream model: ~4GB VRAM
  - LLaMA model: ~6GB VRAM
  - Peak usage during caption generation

## Dependencies
Required Python packages:
- transformers
- torch
- opencv-python (cv2)
- Pillow (PIL)
- numpy
- pyvips
- huggingface-hub
- accelerate
- einops
- tqdm

System requirements:
- FFmpeg
- libvips

All Python dependencies can be installed via:
```bash
pip install -r requirements.txt
```

## Model Details
- **Frame Analysis**: Moondream 2B (2025-01-09 release)
  - Specialized for detailed image understanding
  - Optimized for frame-by-frame analysis

- **Scene Analysis**: Meta's LLaMA 3.2 3B Instruct
  - Used for scene transition detection
  - Generates flowing narratives for scenes
  - Filters and refines captions

## License
This project is licensed under the MIT License.