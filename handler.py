import runpod
import os
import base64
import uuid
from pathlib import Path
from fast_flashtalk import Audio, FlashTalkPipeline, Image

TEMP_DIR = Path("/tmp")  # RunPod temp storage
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- GLOBAL INITIALIZATION ---
# Initialize the 14B model once per worker to avoid reloading on every request.
# Paths are configured via environment variables with fallbacks to a /workspace mount.
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/workspace/SoulX-FlashTalk-14B")
WAV2VEC_DIR = os.environ.get("WAV2VEC_DIR", "/workspace/chinese-wav2vec2-base")

print("Initializing FlashTalkPipeline. This may take a moment...")
pipeline = FlashTalkPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    wav2vec_dir=WAV2VEC_DIR,
    num_persistent_param_in_dit=30_000_000_000,
)
print("Pipeline initialized successfully.")


def save_base64_file(base64_str, filename):
    data = base64.b64decode(base64_str)
    file_path = TEMP_DIR / filename
    with open(file_path, "wb") as f:
        f.write(data)
    return str(file_path)


def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up: {file_path}")
        except Exception as e:
            print(f"Failed to cleanup {file_path}: {e}")


def handler(event):
    print("Worker Start")
    input_data = event['input']
    
    # Track files for cleanup
    temp_files = []
    output_video_path = None

    try:
        # Extract inputs
        photo_b64 = input_data.get("photo")
        audio_b64 = input_data.get("audio")
        # Added a prompt field with a generic default fallback
        input_prompt = input_data.get("input_prompt", "A high-quality talking portrait.")

        if not photo_b64 or not audio_b64:
            return {"error": "Missing 'photo' or 'audio' in input."}

        # Save input files
        image_filename = f"{uuid.uuid4().hex}_photo.png"
        audio_filename = f"{uuid.uuid4().hex}_audio.wav"
        
        image_path = save_base64_file(photo_b64, image_filename)
        audio_path = save_base64_file(audio_b64, audio_filename)
        
        # Track input files for cleanup
        temp_files.extend([image_path, audio_path])
        print("Saved input files.")

        # Setup fast_flashtalk objects
        image = Image(uri=image_path)
        audio = Audio(uri=audio_path)
        
        # Generate output path and track it
        output_video_path = str(TEMP_DIR / f"{uuid.uuid4().hex}_output.mp4")
        temp_files.append(output_video_path)

        # Run inference
        print("Starting video generation...")
        video = pipeline.generate(
            input_prompt=input_prompt,
            audio=audio,
            image=image,
        )
        
        print("Merging audio...")
        video.merge_audio(audio, output_path=output_video_path)
        
        print(f"Generated and merged video at: {output_video_path}")

        # Encode output to base64 to return
        with open(output_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video_base64": video_b64,
            "message": "Inference completed successfully."
        }
        
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}
    
    finally:
        # Always cleanup temp files (even if there was an error)
        cleanup_temp_files(*temp_files)
        print("Worker Finish: Cleanup completed.")


runpod.serverless.start({"handler": handler})