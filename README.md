# Video Captioning with Moondream

> **⚠️ IMPORTANT:** This project uses Moondream 2B (2025-01-09 release) via the Hugging Face Transformers library.

> **💡 NOTE:** This project offers two options for the LLaMA model:
> 1. Local Ollama LLaMA (Recommended)
> 2. HuggingFace LLaMA (Requires approval)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [System Dependencies](#system-dependencies)
  - [Python Dependencies](#python-dependencies)
  - [Model Setup](#model-setup)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

A Python script that automatically generates captions for videos using Moondream for frame analysis and LLaMA for scene detection. The script processes videos frame by frame, detects scene changes, and overlays captions directly onto the video.

## Features

- **Intelligent Frame Extraction**:
  - Time-based sampling (e.g., one frame every N seconds)
  - Fixed total frame count (evenly spaced throughout video)

- **Smart Scene Detection**:
  - Automatic scene change detection using word overlap analysis
  - Detects major content changes while being resilient to minor variations
  - Scene transitions determined when word overlap falls below 33%

- **Structured Caption Generation**:
  - Detailed scene descriptions focusing on:
    1. People present and their actions
    2. Setting and location
    3. Notable objects and activities
    4. Overall atmosphere and mood
  - Real-time streaming of captions during processing

- **Professional Caption Overlay**:
  - Dynamic font sizing based on video dimensions
  - Smart text wrapping (max 3 lines)
  - Semi-transparent background for readability
  - Centered positioning with proper padding
  - Anti-aliased text with thin outline for clarity

## Prerequisites

- Python 3.8 or later
- CUDA-capable GPU (recommended)
- FFmpeg installed
- For LLaMA model access:
  - Either:
    1. Ollama installed locally (recommended)
    2. HuggingFace account with approved access to Meta's LLaMA model

## Installation

### System Dependencies
```bash
# Linux/macOS
sudo apt-get update
sudo apt-get install ffmpeg

# Windows
# Download and install FFmpeg from https://ffmpeg.org/download.html
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Model Setup
- **Option 1 (Recommended)**: Local Ollama
  ```bash
  # The script will automatically:
  # 1. Install Ollama if not present
  # 2. Start the Ollama service
  # 3. Pull the LLaMA model
  ```

- **Option 2**: HuggingFace
  1. Visit [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  2. Request access and wait for approval
  3. Authenticate using one of these methods:
     ```bash
     # Method 1: CLI login (Recommended)
     huggingface-cli login

     # Method 2: Use token
     python video-captioning.py --token "your_token"
     ```

## Usage

1. Place your video files in the `inputs` folder
2. Run the script:
   ```bash
   # Default: Extract one frame every second
   python video-captioning.py

   # Custom frame interval (e.g., every 5 seconds)
   python video-captioning.py --frame-interval 5

   # Fixed number of total frames
   python video-captioning.py --total-frames 30
   ```

## Output

- **Console Output**:
  - Frame extraction progress and timing
  - Real-time caption generation
  - Scene detection analysis
  - Scene-by-scene summary with timestamps

- **Video Output**:
  - Saved in `outputs` folder as `captioned_[original_name].mp4`
  - Original video with overlaid captions
  - Captions change at detected scene boundaries
  - Professional text rendering with dynamic sizing

## Troubleshooting

- **CUDA/GPU Issues**:
  - Ensure CUDA toolkit is installed
  - Check GPU memory usage
  - Try reducing batch size if out of memory

- **Model Loading**:
  - For Ollama: Check if service is running (`http://localhost:11434`)
  - For HuggingFace: Verify model access and authentication

- **Video Processing**:
  - Ensure FFmpeg is properly installed
  - Check input video format compatibility
  - Verify sufficient disk space for frame extraction

## Performance Notes

- Processing time depends on:
  - Video length and resolution
  - Frame extraction interval
  - GPU capabilities
  - Scene complexity

## Dependencies

- `transformers`: Moondream model and LLaMA pipeline
- `torch`: Deep learning backend
- `opencv-python`: Video processing and caption overlay
- `Pillow`: Image handling
- `huggingface_hub`: Model access and authentication
- `requests`: API communication for Ollama

## License

This project is licensed under the MIT License - see the LICENSE file for details.