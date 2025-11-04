import os
import cv2
import numpy as np
import tensorflow as tf

# ✅ Settings
image_height, image_width = 64, 64  # must match model input

# ✅ Model path
model_path = r"C:\Users\AK\Desktop\Internship_Project\Model___Date_Time_2025_10_14__09_34_41___Loss_0.0007586043793708086___Accuracy_1.0 (1).h5"  # simple name recommended

# ✅ Check if model exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found! Please check the path:\n{model_path}")

# ✅ Load trained model
print("✅ Loading trained model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# ✅ Define class labels
class_labels = ["PushUps", "JumpingJack", "TaiChi", "Swing"]

# ✅ Function to extract the middle frame from a video
def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("❌ Could not read frame from video.")

    frame = cv2.resize(frame, (image_width, image_height))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

# ✅ Prediction function
def predict_action(video_path):
    print(f"🎬 Predicting for: {video_path}")
    frame = extract_middle_frame(video_path)
    preds = model.predict(frame)

    top_idx = np.argmax(preds[0])
    confidence = preds[0][top_idx]
    class_name = class_labels[top_idx]

    print("\n🎯 Top Prediction:")
    print(f"1. {class_name} — {confidence:.4f}")

# ✅ Run prediction on local video
if __name__ == "__main__":
    video_path = r"C:\Users\AK\Desktop\Internship_Project\How To Do A Plank To Push Up.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video file not found! Please check the path:\n{video_path}")
    
    predict_action(video_path)
