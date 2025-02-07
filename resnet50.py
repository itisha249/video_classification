from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
import os
import torch
import numpy as np

# Load the processor and model (choose one)
# CLIP
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# ResNet-50
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video at a specified frame rate.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Invalid FPS value {fps}. Check the video file.")

    frame_interval = max(1, int(fps // frame_rate))
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

def classify_video(video_path):
    """
    Classify a video scene as "Movie Scene" or "Real-Life Scene".
    """
    frame_folder = "frames"
    frames = extract_frames(video_path, frame_folder, frame_rate=1)

    # Text prompts for CLIP (if used)
    text = ["This is a movie scene", "This is a real-life scene"]
    probs_list = []

    for frame_path in frames:
        try:
            image = Image.open(frame_path).convert("RGB")
            # Adjust for ResNet or CLIP
            if isinstance(processor, CLIPProcessor):
                inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()
            else:  # ResNet-50
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()

            probs_list.append(probs)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            continue

    if not probs_list:
        raise ValueError("No valid probabilities computed. Check frame extraction or model inference.")

    # Aggregate probabilities
    avg_probs = np.mean(probs_list, axis=0)
    #print(f"Aggregated probabilities: {avg_probs}")

   # Format percentages
    movie_scene_prob = avg_probs[0][0] * 1000
    real_life_scene_prob = avg_probs[0][1] * 1000

    print(f"Aggregated probabilities: Movie Scene: {movie_scene_prob}, Real-Life Scene: {real_life_scene_prob}")

    return ("Movie Scene :" ,{movie_scene_prob}, ", Real-Life Scene:",{real_life_scene_prob})

try:
    result = classify_video(r"D:\video_classification\video_classification\fake.mp4")  # Replace with your video path
    print(result)
except Exception as e:
    print(f"Error: {e}")