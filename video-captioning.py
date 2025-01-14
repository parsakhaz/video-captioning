import cv2
import os
import platform
import subprocess
import sys
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Union
import warnings
import argparse
from huggingface_hub import login, whoami
import requests
import json
import tempfile
import shutil
from pathlib import Path
import time
warnings.filterwarnings('ignore')

# Define paths
INPUT_FOLDER = "./inputs"
OUTPUT_FOLDER = "vidframes"

# Ollama wrapper class for consistent API
class OllamaWrapper:
    def __call__(self, messages, **kwargs):
        prompt = messages[1]["content"]  # Get user content from messages
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": f"Summarize and arrange these image captions into one flowing narrative, return the narrative only, do not include any other text: {prompt}",
                "stream": False
            }
        )
        return [{"generated_text": response.json()["response"]}]

def get_os_type():
    """Determine the OS type."""
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    elif system == "windows":
        return "windows"
    else:
        return "linux"

def install_ollama():
    """Install Ollama based on the OS."""
    os_type = get_os_type()
    print(f"\nDetected OS: {os_type.capitalize()}")
    print("Attempting to install Ollama...")

    try:
        if os_type == "mac":
            # macOS installation using Homebrew
            try:
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
            except:
                print("Homebrew not found. Installing Homebrew...")
                subprocess.run(['/bin/bash', '-c', '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'])
            
            print("Installing Ollama via Homebrew...")
            subprocess.run(["brew", "install", "ollama"])

        elif os_type == "linux":
            # Linux installation using curl
            print("Installing Ollama via curl...")
            install_cmd = 'curl https://ollama.ai/install.sh | sh'
            subprocess.run(install_cmd, shell=True, check=True)

        elif os_type == "windows":
            # Windows installation using official MSI
            print("Downloading Ollama installer...")
            temp_dir = tempfile.mkdtemp()
            installer_path = os.path.join(temp_dir, "ollama-installer.msi")
            
            response = requests.get("https://ollama.ai/download/windows", stream=True)
            with open(installer_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            print("Installing Ollama...")
            subprocess.run(['msiexec', '/i', installer_path, '/quiet'], check=True)
            shutil.rmtree(temp_dir)

        print("Ollama installation completed!")
        
    except Exception as e:
        print(f"\nError installing Ollama: {str(e)}")
        print("\nPlease install Ollama manually:")
        print("1. Visit https://ollama.ai")
        print("2. Download and install the appropriate version for your OS")
        print("3. Run the script again after installation")
        raise Exception("Ollama installation failed")

def start_ollama_service():
    """Start the Ollama service based on OS."""
    os_type = get_os_type()
    
    try:
        if os_type == "windows":
            # Check if service is running
            try:
                requests.get("http://localhost:11434/api/tags")
                return  # Service is running
            except:
                # Start Ollama service
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux and Mac
            try:
                requests.get("http://localhost:11434/api/tags")
                return  # Service is running
            except:
                # Start Ollama service
                subprocess.Popen(['ollama', 'serve'])
        
        # Wait for service to start
        print("Starting Ollama service...")
        max_retries = 10
        for i in range(max_retries):
            try:
                requests.get("http://localhost:11434/api/tags")
                print("Ollama service started successfully!")
                return
            except:
                if i < max_retries - 1:
                    print(f"Waiting for service to start... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    raise Exception("Service failed to start")
                
    except Exception as e:
        print(f"\nError starting Ollama service: {str(e)}")
        print("Please start Ollama manually and try again")
        raise

def pull_llama_model():
    """Pull the LLaMA model in Ollama."""
    print("\nPulling LLaMA model...")
    try:
        subprocess.run(['ollama', 'pull', 'llama3.2:1b'], check=True)
        print("LLaMA model pulled successfully!")
    except Exception as e:
        print(f"\nError pulling LLaMA model: {str(e)}")
        raise

def setup_ollama():
    """Setup and verify Ollama LLaMA."""
    print("\nChecking Ollama setup...")
    
    try:
        # Check if Ollama is installed
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except:
        install_ollama()
    
    # Start Ollama service
    start_ollama_service()
    
    # Check if model exists
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models = response.json()
        if not any(model["name"].startswith("llama3.2:1b") for model in models["models"]):
            pull_llama_model()
    else:
        pull_llama_model()
    
    print("Ollama LLaMA model ready!")

def get_model_choice():
    """Let user choose between HuggingFace and Ollama LLaMA."""
    print("\nChoose LLaMA model source:")
    print("1. Local Ollama LLaMA (Recommended, requires Ollama installed)")
    print("2. HuggingFace LLaMA (Requires approved access)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")

def setup_authentication(token: str = None):
    """Setup HuggingFace authentication either via web UI or token."""
    print("\nChecking HuggingFace authentication...")
    print("This is required to access the LLaMA model for narrative generation.")
    print("Note: You must have been granted access to Meta's LLaMA model.")
    print("      If you haven't requested access yet, visit:")
    print("      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    
    try:
        if token:
            print("Using provided HuggingFace token...")
            login(token=token)
        else:
            try:
                # Try to get existing login
                user_info = whoami()
                print(f"Found existing login as: {user_info['name']} ({user_info['email']})")
                return
            except Exception:
                print("\nNo existing login found. You can:")
                print("1. Pre-authenticate using the command line:")
                print("   $ huggingface-cli login")
                print("2. Run this script with a token:")
                print("   $ python video-captioning.py --token YOUR_TOKEN")
                print("\nPlease authenticate using one of these methods and try again.")
                raise Exception("No authentication found. Please authenticate using one of the methods above.")
        
        # Verify authentication
        user_info = whoami()
        print(f"\nSuccessfully authenticated as: {user_info['name']} ({user_info['email']})")
        
    except Exception as e:
        print("\nAuthentication Error:")
        print("Make sure you have:")
        print("1. A HuggingFace account")
        print("2. Requested and been granted access to Meta's LLaMA model")
        print("   Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("   Note: The approval process may take several days")
        print("3. Either:")
        print("   - Run 'huggingface-cli login' in your terminal")
        print("   - Provide a valid token with --token")
        
        if "Cannot access gated repo" in str(e) or "awaiting a review" in str(e):
            print("\nError: You don't have access to the LLaMA model yet.")
            print("Please request access and wait for approval before using this script.")
            print("Alternatively, you can use the local Ollama LLaMA option.")
        else:
            print("\nError details:", str(e))
        raise

def check_ollama():
    """Check if Ollama is running and has LLaMA model."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            return False
        
        models = response.json()
        return any(model["name"].startswith("llama3.2:1b") for model in models["models"])
    except:
        return False

def extract_frames(
    video_path: Union[str, os.PathLike],
    output_folder: str = OUTPUT_FOLDER,
    frame_interval: float = 1.0,
    total_frames: int = None
) -> tuple:
    """Extract frames and record their timestamps."""
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / fps
    
    if total_frames:
        # Calculate frame interval to get desired number of frames
        frame_interval_frames = max(1, total_video_frames // total_frames)
        print(f"Video duration: {duration:.1f}s, extracting {total_frames} frames...")
    else:
        # Use time-based sampling
        frame_interval_frames = int(fps * frame_interval)
        estimated_frames = total_video_frames // frame_interval_frames
        print(f"Video duration: {duration:.1f}s, extracting ~{estimated_frames} frames (one every {frame_interval}s)...")
    
    frame_count = 0
    frames_saved = 0
    timestamps = []
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % frame_interval_frames == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            timestamps.append(frame_count / fps)
            frames_saved += 1
        
        frame_count += 1
    
    video.release()
    print(f"Extracted {frames_saved} frames from video")
    return output_folder, timestamps

def load_models(use_ollama: bool = False) -> tuple:
    """Load the Moondream and LLama models."""
    print('Loading models...')
    
    # Load Moondream model with latest revision
    moondream_model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        # Uncomment to run on GPU
        device_map={"": "cuda"}
    )
    
    if use_ollama:
        # Return an instance of OllamaWrapper
        return moondream_model, OllamaWrapper()
    else:
        # Load HuggingFace LLaMA pipeline
        llm_pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return moondream_model, llm_pipe

def caption_frames(
    image_frames: list,
    model: AutoModelForCausalLM,
) -> list:
    """Generate captions for each frame using Moondream API."""
    captions = []
    
    structured_prompt = """Describe this image in detail, focusing on:
1. Any people present and what they're doing
2. The setting or location
3. Notable objects or activities
4. The overall atmosphere or mood"""
    
    for i, frame in enumerate(image_frames):
        print(f"\nFrame {i+1}/{len(image_frames)}:")
        print("-" * (len(str(i+1)) + len(str(len(image_frames))) + 8))
        
        # Stream caption
        print("Caption: ", end="", flush=True)
        caption = ""
        for t in model.query(frame, structured_prompt, stream=True)["answer"]:
            print(t, end="", flush=True)
            caption += t
        
        captions.append(caption)
        print("\n" + "-" * 50)
        
    return captions

def add_caption_to_frame(frame, caption):
    """Add caption to frame with smart sizing and positioning."""
    height, width = frame.shape[:2]
    
    # Calculate base font size (proportional to video height)
    base_font_size = height / 30  # Adjust this ratio for different sizes
    
    # Initialize font size
    font_size = base_font_size
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Calculate maximum width for text (70% of frame width)
    max_width = int(width * 0.7)
    
    # Function to wrap text
    def wrap_text(text, max_width, font_size):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_size = cv2.getTextSize(word + " ", font, font_size / base_font_size, 1)[0]
            if current_width + word_size[0] <= max_width:
                current_line.append(word)
                current_width += word_size[0]
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_size[0]
        
        if current_line:
            lines.append(" ".join(current_line))
        return lines
    
    # Adjust font size if text is too wide
    while font_size > base_font_size / 2:  # Don't go smaller than half base size
        lines = wrap_text(caption, max_width, font_size)
        if len(lines) <= 3:  # Maximum 3 lines
            break
        font_size *= 0.9
    
    # Calculate text dimensions
    line_height = int(font_size * 1.5)
    total_height = line_height * len(lines)
    
    # Calculate position (centered, near bottom with padding)
    bottom_padding = height // 15  # Padding from bottom
    start_y = height - bottom_padding - total_height
    
    # Draw semi-transparent background
    overlay = frame.copy()
    bg_padding = int(font_size / 2)  # Padding around text
    bg_start_y = start_y - bg_padding
    bg_height = total_height + 2 * bg_padding
    cv2.rectangle(overlay, 
                 (0, bg_start_y), 
                 (width, bg_start_y + bg_height),
                 (0, 0, 0), 
                 -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw each line of text
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_size / base_font_size, 1)[0]
        x = (width - text_size[0]) // 2  # Center horizontally
        y = start_y + (i + 1) * line_height
        
        # Draw text with thin black outline for better readability
        cv2.putText(frame, line, (x, y), font, font_size / base_font_size, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, font_size / base_font_size, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def analyze_scene_transition(caption1: str, caption2: str) -> tuple[bool, str]:
    """Compare two captions to detect scene transitions using word overlap percentage.
    A word overlap below 33% indicates a scene change."""
    
    # Convert captions to word sets
    words1 = set(caption1.lower().split())
    words2 = set(caption2.lower().split())
    
    # Calculate word overlap percentage
    common_words = words1 & words2
    overlap_percentage = len(common_words) / max(len(words1), len(words2))
    
    # Scene transition if overlap is less than 33%
    is_transition = overlap_percentage < 0.33
    
    return is_transition, ""

def group_captions_into_segments(captions: list, frame_timestamps: list, llm_pipeline) -> list:
    """Group captions into segments based on word differences between consecutive captions.
    Uses the first frame's caption for each scene."""
    print("\nAnalyzing scene transitions...")
    segments = []
    current_segment = {
        "captions": [captions[0]],  # We'll only use this first caption
        "start_time": frame_timestamps[0],
        "end_time": frame_timestamps[1] if len(frame_timestamps) > 1 else frame_timestamps[0],
        "caption": captions[0]  # Use first frame's caption directly
    }
    
    for i in range(1, len(captions)):
        print(f"\nAnalyzing frames {i}/{len(captions)-1}...")
        is_transition, _ = analyze_scene_transition(captions[i-1], captions[i])
        
        if is_transition:
            # Add current segment
            segments.append(current_segment)
            
            # Print segment information
            print(f"\nScene {len(segments)}:")
            print(f"Time: {current_segment['start_time']:.1f}s - {current_segment['end_time']:.1f}s")
            print(f"Caption: {current_segment['caption']}")
            
            # Start new segment with first frame of new scene
            current_segment = {
                "captions": [captions[i]],  # We'll only use this first caption
                "start_time": frame_timestamps[i],
                "end_time": frame_timestamps[i+1] if i+1 < len(frame_timestamps) else frame_timestamps[-1],
                "caption": captions[i]  # Use first frame's caption directly
            }
        else:
            # Just update end time of current segment
            current_segment["end_time"] = frame_timestamps[i+1] if i+1 < len(frame_timestamps) else frame_timestamps[-1]
    
    # Add final segment
    segments.append(current_segment)
    print(f"\nFinal Scene:")
    print(f"Time: {current_segment['start_time']:.1f}s - {current_segment['end_time']:.1f}s")
    print(f"Caption: {current_segment['caption']}")
    
    # Print summary
    print(f"\nVideo Summary:")
    print(f"Total Scenes: {len(segments)}")
    for i, segment in enumerate(segments, 1):
        print(f"\nScene {i}:")
        print(f"Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")
        print(f"Caption: {segment['caption']}")
    
    return segments

def create_captioned_video(
    input_path: str,
    output_path: str,
    segments: list
) -> None:
    """Create video with captions."""
    print("\nCreating captioned video...")
    
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    frame_count = 0
    current_segment_idx = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        # Find current segment
        while (current_segment_idx < len(segments) - 1 and 
               current_time >= segments[current_segment_idx]["end_time"]):
            current_segment_idx += 1
        
        # Add caption to frame
        frame = add_caption_to_frame(frame, segments[current_segment_idx]["caption"])
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Show progress
        if frame_count % 30 == 0:  # Update progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
    
    # Clean up
    video.release()
    out.release()
    print("\nCaptioned video saved to:", output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Video Captioning with Moondream',
        epilog='If using HuggingFace LLaMA, authentication is required.'
    )
    parser.add_argument('--token', type=str, help='Optional: HuggingFace token for authentication')
    
    # Add frame extraction options
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        '--frame-interval', 
        type=float, 
        default=1.0,
        help='Extract one frame every N seconds (default: 1.0)'
    )
    frame_group.add_argument(
        '--total-frames',
        type=int,
        help='Extract N frames evenly spaced throughout the video'
    )
    
    args = parser.parse_args()

    print("\nVideo Captioning with Moondream")
    print("===============================")
    
    # Get model choice
    model_choice = get_model_choice()
    use_ollama = (model_choice == '1')
    
    # Setup authentication/Ollama
    if use_ollama:
        setup_ollama()
    else:
        setup_authentication(args.token)
    
    # Get all video files from input folder
    print("\nChecking input folder...")
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("\nNo video files found in 'inputs' folder!")
        print("Please add your videos to the 'inputs' folder and run the script again.")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        return
        
    # Process each video
    print(f"\nFound {len(video_files)} video(s) to process.")
    
    for video_file in video_files:
        video_path = os.path.join(INPUT_FOLDER, video_file)
        print(f"\nProcessing video: {video_file}")
        print("=" * (len(video_file) + 16))
        
        # 1. Extract frames
        print("\n1. Extracting frames from video...")
        output_folder, timestamps = extract_frames(
            video_path,
            frame_interval=args.frame_interval,
            total_frames=args.total_frames
        )
        
        # 2. Load frames
        print("\n2. Loading extracted frames...")
        vidframes = [os.path.join(output_folder, path) for path in os.listdir(output_folder)]
        vidframes.sort()  # Ensure frames are in order
        image_frames = [Image.open(img) for img in vidframes]
        
        # 3. Load models
        print("\n3. Loading AI models...")
        print("   This may take a moment depending on your internet speed.")
        moondream_model, llm_pipe = load_models(use_ollama)
        
        # 4. Generate initial captions
        print("\n4. Generating initial captions...")
        captions = caption_frames(image_frames, moondream_model)
        
        # 5. Analyze scenes and group into segments
        print("\n5. Analyzing scenes and creating segments...")
        segments = group_captions_into_segments(captions, timestamps, llm_pipe)
        
        # 6. Create final video with segmented captions
        print("\n6. Creating final video...")
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"captioned_{video_file}")
        create_captioned_video(
            video_path,
            output_path,
            segments
        )
        
        # Clean up frames
        print("\nCleaning up temporary files...")
        for frame in image_frames:
            frame.close()
        for frame_path in vidframes:
            os.remove(frame_path)
        
        print("\nProcessing complete!")

if __name__ == "__main__":
    main()