import cv2
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Union, List
import warnings
import argparse
from huggingface_hub import login, whoami
import json
import shutil
import gc
from datetime import datetime
import re

# Configure PyTorch memory allocation
torch.cuda.empty_cache()
gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clean up any existing frames at startup
if os.path.exists("vidframes"):
    shutil.rmtree("vidframes")

warnings.filterwarnings('ignore')

# Define paths
INPUT_FOLDER = "./inputs"
OUTPUT_FOLDER = "vidframes"

def setup_authentication(token: str = None):
    """Setup HuggingFace authentication either via web UI or token."""
    print("\nChecking HuggingFace authentication...")
    print("This is required to access the LLaMA model for narrative generation.")
    print("Note: You must have been granted access to Meta's LLaMA model.")
    print("      If you haven't requested access yet, visit:")
    print("      https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    
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
        print("   Visit: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        print("   Note: The approval process may take several days")
        print("3. Either:")
        print("   - Run 'huggingface-cli login' in your terminal")
        print("   - Provide a valid token with --token")
        
        if "Cannot access gated repo" in str(e) or "awaiting a review" in str(e):
            print("\nError: You don't have access to the LLaMA model yet.")
            print("Please request access and wait for approval before using this script.")
        raise

def extract_frames(
    video_path: Union[str, os.PathLike],
    output_folder: str = OUTPUT_FOLDER,
    frame_interval: float = 0.67,
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

def load_moondream() -> AutoModelForCausalLM:
    """Load just the Moondream model."""
    print('Loading Moondream model...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).cuda()
    return model

def load_llama():
    """Load the LLaMA model."""
    print('Loading LLaMA model...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

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

def refine_captions(captions: List[str], llm_pipeline) -> str:
    """Generate a flowing narrative from multiple captions of the same scene."""
    # Format captions with timestamps for better context
    formatted_captions = []
    for i, caption in enumerate(captions):
        formatted_captions.append(f"Frame {i+1}: {caption}")
    
    scene_description = "\n".join(formatted_captions)
    
    messages = [
        {
            "role": "system",
            "content": "You are a video editor describing a single scene from its frames. You only respond with the detailed description of the entire scene. Start your description with 'A' or 'The' and describe exactly what is happening in the scene from the frame descriptions. Provide 4-5 detailed sentences, focusing on aggregating details that are mentioned in the frame descriptions. Do not add any details that aren't directly stated. Return a single clear description. No extra text."
        },
        {
            "role": "user", 
            "content": f"These are descriptions of a single scene from different frames:\n\n{scene_description}\n\nProvide a clear description in 4-5 sentences that captures confidently what is happening in this scene. Infer and keep only the correct details. Do not add any details that aren't directly stated. Return a single assertive clear description. No extra text. No concluding sentence that starts with 'Overall ...' - only the very matter of fact scene description."
        }
    ]

    # Generate caption using HuggingFace pipeline
    outputs = llm_pipeline(
        messages,
        max_new_tokens=512,  # Increased token limit
        do_sample=True,
        temperature=0.05,  # Much lower temperature
        top_p=0.1,  # Much more conservative top_p
        repetition_penalty=1.2,
        pad_token_id=2
    )
    return outputs[0]["generated_text"][-1]["content"].strip()

def filter_scene_captions(captions: List[str], llm_pipeline) -> tuple[List[str], dict]:
    """Filter scene captions to identify core recurring elements across frames."""
    # Format captions with timestamps for better context
    formatted_captions = []
    for i, caption in enumerate(captions):
        formatted_captions.append(f"Frame {i+1}: {caption}")
    
    scene_description = "\n".join(formatted_captions)
    
    messages = [
        {
            "role": "system",
            "content": "You analyze multiple descriptions of the same scene and identify the core elements that appear consistently. Return only a JSON object with the key elements."
        },
        {
            "role": "user", 
            "content": f"""These are descriptions of the same scene from different frames:

{scene_description}

Return a JSON object with these fields:
- main_subject: The primary subject/person/object that appears most consistently
- action: The main action or activity that is described repeatedly
- setting: The consistent location or setting mentioned
- key_details: List of 2-3 other details that appear in multiple frames
- visible_text: List of any text, words, numbers, or signs that appear in the scene repeatedly

Return ONLY the JSON object, no other text."""
        }
    ]

    # Generate analysis using HuggingFace pipeline
    outputs = llm_pipeline(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.05,
        top_p=0.1,
        repetition_penalty=1.2,
        pad_token_id=2
    )
    
    # Get the raw JSON string
    raw_json = outputs[0]["generated_text"][-1]["content"].strip()
    
    try:
        # Try to parse the JSON response
        analysis = json.loads(raw_json)
        filtered_caption = f"A {analysis['main_subject']} {analysis['action']} in {analysis['setting']}. "
        if analysis['key_details']:
            filtered_caption += " ".join(analysis['key_details']) + "."
        # Add any visible text
        if analysis.get('visible_text') and analysis['visible_text']:
            text_items = [f'"{text}"' for text in analysis['visible_text']]
            filtered_caption += f" The text {text_items[0] if len(text_items) == 1 else ' and '.join(text_items)} is visible in the scene."
    except:
        # If parsing fails, use the raw JSON string
        filtered_caption = raw_json
        analysis = "JSON parsing failed, using raw output"
    
    return [filtered_caption], {
        "raw_captions": captions,
        "filtered_captions": [filtered_caption],
        "analysis": analysis
    }

def group_captions_into_segments(captions: list, frame_timestamps: list, llm_pipeline, video_file: str) -> list:
    """Group captions into segments based on word differences between consecutive captions.
    Uses LLaMA to generate flowing narratives for each scene."""
    print("\nAnalyzing scene transitions...")
    segments = []
    current_segment = {
        "captions": [captions[0]],
        "start_time": frame_timestamps[0],
        "end_time": frame_timestamps[1] if len(frame_timestamps) > 1 else frame_timestamps[0],
        "frame_indices": [0]  # Track frame indices
    }
    
    for i in range(1, len(captions)):
        print(f"\nAnalyzing frames {i}/{len(captions)-1}...")
        is_transition, _ = analyze_scene_transition(captions[i-1], captions[i])
        
        if is_transition:
            # Filter captions to remove hallucinations
            print("\nFiltering scene captions...")
            filtered_captions, analysis = filter_scene_captions(current_segment["captions"], llm_pipeline)
            
            # Generate refined caption for current segment using filtered captions
            print("\nGenerating flowing narrative for scene...")
            refined_caption = refine_captions(filtered_captions, llm_pipeline)
            
            # Store all information
            current_segment["caption"] = refined_caption
            current_segment["caption_analysis"] = analysis
            
            # Add current segment
            segments.append(current_segment)
            
            # Print segment information
            print(f"\nScene {len(segments)}:")
            print(f"Time: {current_segment['start_time']:.1f}s - {current_segment['end_time']:.1f}s")
            print(f"Caption: {current_segment['caption']}")
            
            # Start new segment
            current_segment = {
                "captions": [captions[i]],
                "start_time": frame_timestamps[i],
                "end_time": frame_timestamps[i+1] if i+1 < len(frame_timestamps) else frame_timestamps[-1],
                "frame_indices": [i]
            }
        else:
            # Add caption to current segment
            current_segment["captions"].append(captions[i])
            current_segment["frame_indices"].append(i)
            current_segment["end_time"] = frame_timestamps[i+1] if i+1 < len(frame_timestamps) else frame_timestamps[-1]
    
    # Process final segment
    print("\nFiltering final scene captions...")
    filtered_captions, analysis = filter_scene_captions(current_segment["captions"], llm_pipeline)
    
    print("\nGenerating flowing narrative for final scene...")
    refined_caption = refine_captions(filtered_captions, llm_pipeline)
    current_segment["caption"] = refined_caption
    current_segment["caption_analysis"] = analysis
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
    
    # Generate and save JSON output
    json_output = {
        "video_file": video_file,
        "analysis_timestamp": datetime.now().isoformat(),
        "total_scenes": len(segments),
        "scenes": []
    }
    
    for i, segment in enumerate(segments, 1):
        scene_data = {
            "scene_number": i,
            "time_range": {
                "start": f"{segment['start_time']:.1f}s",
                "end": f"{segment['end_time']:.1f}s"
            },
            "final_caption": segment['caption'],
            "caption_analysis": {
                "recurring_elements": segment['caption_analysis']['analysis'],
                "raw_captions": segment['caption_analysis']['raw_captions'],
                "filtered_captions": segment['caption_analysis']['filtered_captions']
            },
            "frames": []
        }
        
        for idx, (frame_idx, frame_caption) in enumerate(zip(segment['frame_indices'], segment['captions'])):
            scene_data["frames"].append({
                "frame_number": frame_idx,
                "timestamp": f"{frame_timestamps[frame_idx]:.1f}s",
                "original_caption": frame_caption
            })
        
        json_output["scenes"].append(scene_data)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Normalize filename by removing extension and special characters
    base_filename = os.path.splitext(video_file)[0]
    normalized_filename = re.sub(r'[^\w\s-]', '', base_filename).strip().lower()
    normalized_filename = re.sub(r'[-\s]+', '-', normalized_filename)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"scene_analysis_{normalized_filename}_{timestamp}.json"
    json_path = os.path.join("logs", json_filename)
    
    # Save JSON to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed scene analysis saved to: {json_path}")
    
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
        epilog='HuggingFace authentication is required.'
    )
    parser.add_argument('--token', type=str, help='Optional: HuggingFace token for authentication')
    
    # Add frame extraction options
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        '--frame-interval', 
        type=float, 
        default=0.67,
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
    
    # Setup authentication
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
    
    # Clean up any existing frames
    if os.path.exists(OUTPUT_FOLDER):
        print("\nCleaning up existing frames...")
        shutil.rmtree(OUTPUT_FOLDER)
    
    for video_file in video_files:
        try:
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
        
            # 3. Load Moondream and generate captions
            print("\n3. Loading Moondream model...")
            moondream_model = None
            captions = []
            try:
                moondream_model = load_moondream()
                print("\n4. Generating initial captions...")
                captions = caption_frames(image_frames, moondream_model)
            finally:
                # Clean up Moondream
                if moondream_model:
                    del moondream_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # 4. Load LLaMA and process scenes
            print("\n5. Loading LLaMA model...")
            llm_pipe = None
            segments = []
            try:
                llm_pipe = load_llama()
                print("\n6. Analyzing scenes and creating segments...")
                segments = group_captions_into_segments(captions, timestamps, llm_pipe, video_file)
            finally:
                # Clean up LLaMA
                if llm_pipe:
                    del llm_pipe
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # 5. Create final video
            print("\n7. Creating final video...")
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
                try:
                    os.remove(frame_path)
                except Exception as e:
                    print(f"Warning: Could not remove {frame_path}: {e}")
            
            print("\nProcessing complete!")
            
        except Exception as e:
            print(f"\nError processing {video_file}: {str(e)}")
            continue
        
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()