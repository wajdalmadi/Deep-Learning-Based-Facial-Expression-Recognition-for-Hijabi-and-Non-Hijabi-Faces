"""
Live Facial Emotion Recognition
Real-time emotion detection using ResNet18 and OpenCV
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from collections import deque
import os

print("="*70)
print(" Live Facial Emotion Recognition")
print("="*70)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'neutral']
IMG_SIZE = 224
MODEL_PATH = 'best_baseline_model.pth'  # Make sure model is in the same folder

# Colors for emotions (BGR format for OpenCV)
EMOTION_COLORS = {
    'happy': (0, 255, 0),       # Green
    'sad': (255, 0, 0),         # Blue
    'angry': (0, 0, 255),       # Red
    'surprise': (0, 255, 255),  # Yellow
    'neutral': (128, 128, 128)  # Gray
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1. Check if model exists
# ═══════════════════════════════════════════════════════════════

if not os.path.exists(MODEL_PATH):
    print(f" Error: Model not found at: {MODEL_PATH}")
    print(f"   Make sure you downloaded it from Colab and placed it in the same folder")
    exit()

print(f" Model found: {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════════
# 2. Load model
# ═══════════════════════════════════════════════════════════════

print("\n Loading model...")

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(EMOTIONS))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(" Model ready!")

# ═══════════════════════════════════════════════════════════════
# 3. Image transform pipeline
# ═══════════════════════════════════════════════════════════════

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════════
# 4. Face detection setup
# ═══════════════════════════════════════════════════════════════

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print(" Face detector ready!")

# ═══════════════════════════════════════════════════════════════
# 5. Prediction smoothing to reduce flicker
# ═══════════════════════════════════════════════════════════════

class PredictionSmoother:
    """Smooth predictions over time using a sliding window"""
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
    
    def update(self, pred):
        """Add new prediction and return most common recent prediction"""
        self.predictions.append(pred)
        if len(self.predictions) >= 3:
            return max(set(self.predictions), key=self.predictions.count)
        return pred
    
    def reset(self):
        """Clear prediction history"""
        self.predictions.clear()

# ═══════════════════════════════════════════════════════════════
# 6. Emotion prediction function
# ═══════════════════════════════════════════════════════════════

def predict_emotion(face_img):
    """
    Predict emotion from face image
    Returns: (emotion_name, confidence_score)
    """
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Apply transforms
        face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        emotion_idx = predicted.item()
        emotion = EMOTIONS[emotion_idx]
        conf = confidence.item()
        
        return emotion, conf
        
    except Exception as e:
        print(f" Prediction error: {e}")
        return "neutral", 0.0

# ═══════════════════════════════════════════════════════════════
# 7. Main video processing loop
# ═══════════════════════════════════════════════════════════════

print("\n Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot open camera!")
    print("   Try: cv2.VideoCapture(1) if you have multiple cameras")
    exit()

# Set resolution (optional - if running slow)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(" Camera active!")
print("\n" + "="*70)
print("Controls:")
print("  q or ESC → Quit")
print("  s → Save screenshot")
print("  r → Reset smoothing")
print("="*70 + "\n")

smoothers = {}  # Dictionary to store smoothers for each detected face
frame_count = 0
screenshot_count = 0

try:
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print(" Failed to read frame")
            break
        
        frame_count += 1
        
        # Flip for mirror effect (optional)
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        # Draw info bar at top
        info_text = f"Faces: {len(faces)} | Frame: {frame_count}"
        cv2.putText(frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                continue
            
            # Predict emotion
            emotion, confidence = predict_emotion(face)
            
            # Initialize smoother for this face if needed
            if i not in smoothers:
                smoothers[i] = PredictionSmoother(window_size=7)
            
            # Get smoothed prediction
            smoothed_emotion = smoothers[i].update(emotion)
            
            # Get color for this emotion
            color = EMOTION_COLORS.get(smoothed_emotion, (255, 255, 255))
            
            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Prepare emotion text (only emotion name)
            text = smoothed_emotion.upper()
            
            # Draw background for text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x, y-th-10), (x+tw+10, y), color, -1)
            
            # Draw emotion text
            cv2.putText(frame, text, (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Display frame
        cv2.imshow('Live Emotion Recognition - Press Q to Quit', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            print("\n Exiting...")
            break
        elif key == ord('s'):  # Save screenshot
            screenshot_count += 1
            filename = f"emotion_screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f" Saved: {filename}")
        elif key == ord('r'):  # Reset smoothing
            smoothers.clear()
            print(" Reset smoothing")

except KeyboardInterrupt:
    print("\n Interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n Closed successfully!")
    print(f" Total frames processed: {frame_count}")
    if screenshot_count > 0:
        print(f" Screenshots saved: {screenshot_count}")