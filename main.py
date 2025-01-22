from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import os
import torch
import numpy as np

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    # Check if the video file can be opened
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Handle edge cases where FPS might not be available
    if fps <= 0:
        raise ValueError(f"Invalid FPS value {fps}. Check the video file.")

    frame_interval = int(fps // frame_rate)
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)

        frame_count += 1

    cap.release()

    if not saved_frames:
        raise ValueError("No frames were extracted from the video. Check the input video and parameters.")

    return saved_frames


def classify_video_with_clip(video_path):
    """
    Classify a video using CLIP.
    """
    frame_folder = "frames"
    frames = extract_frames(video_path, frame_folder, frame_rate=1)

    text = ["This is a movie scene", "This is a real-life scene"]
    probs_list = []

    for frame_path in frames:
        try:
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()
            probs_list.append(probs)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")

    # Ensure probs_list is not empty before aggregation
    if not probs_list:
        raise ValueError("No valid probabilities computed. Check frame extraction or model inference.")

    # Aggregate probabilities
    avg_probs = np.mean(probs_list, axis=0)
    print(f"Aggregated probabilities: {avg_probs}")  # Debugging output

    return {"Movie Scene": avg_probs[0][0], "Real-Life Scene": avg_probs[0][1]}

# Example usage:
try:
    result = classify_video_with_clip(r"D:\video_classification\video_classification\real_video3.mp4")  # Replace with the actual video path
    print(result)
except Exception as e:
    print(f"Error: {e}")
