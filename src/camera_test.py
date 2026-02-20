"""
Simple real-time camera test for bottle cap classifier.

Controls:
  SPACE = capture and predict on current frame
  C     = continuous mode (auto-predicts every frame)
  Q     = quit

Shows:
  - Live camera feed with preprocessing preview
  - Confidence scores for each class
  - Prediction with color coding
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf


# Custom FocalLoss class (same as in train.py)
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_onehot * tf.math.log(y_pred)
        ce = tf.reduce_sum(ce, axis=-1)
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(self.alpha * focal_weight * ce)


def load_class_names(path: str) -> list:
    """Load class names from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'class_names' in data:
        return data['class_names']
    else:
        raise ValueError(f"Unexpected format in {path}")


def preprocess_frame(frame: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess frame same way as training:
    1. Convert to grayscale
    2. Resize to 224x224
    3. Normalize to 0-1
    4. Add batch dimension
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, target_size)
    
    # Normalize to 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    # Add channel dimension (224, 224) -> (224, 224, 1)
    with_channel = np.expand_dims(normalized, axis=-1)
    
    # Add batch dimension (224, 224, 1) -> (1, 224, 224, 1)
    batch = np.expand_dims(with_channel, axis=0)
    
    return batch, resized


def predict(model: tf.keras.Model, frame_batch: np.ndarray, class_names: list) -> Dict:
    """Run prediction on preprocessed frame."""
    predictions = model.predict(frame_batch, verbose=0)
    probs = predictions[0]
    
    pred_class_idx = np.argmax(probs)
    pred_class = class_names[pred_class_idx]
    confidence = float(probs[pred_class_idx])
    
    return {
        'class': pred_class,
        'confidence': confidence,
        'probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        'all_probs': probs
    }


def get_color(class_name: str) -> tuple:
    """Return BGR color for class visualization."""
    color_map = {
        'good_cap': (0, 255, 0),        # Green
        'loose_cap': (0, 165, 255),     # Orange
        'broken_cap': (0, 0, 255),      # Red
        'broken_ring': (0, 255, 255),   # Yellow
        'no_cap': (128, 0, 128),        # Purple
    }
    return color_map.get(class_name, (255, 255, 255))


def main():
    parser = argparse.ArgumentParser(description='Real-time camera test for cap classifier')
    parser.add_argument('--model', type=str, default='models/cap_classifier_best.keras',
                        help='Path to trained model')
    parser.add_argument('--class_names', type=str, default='models/class_names.json',
                        help='Path to class names JSON')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (0 for built-in)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Only show prediction if confidence > threshold')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={'FocalLoss': FocalLoss}
    )
    print(f"✓ Model loaded. Input shape: {model.input_shape}")
    
    # Load class names
    print(f"Loading class names from {args.class_names}...")
    class_names = load_class_names(args.class_names)
    print(f"✓ Classes: {class_names}")
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    # Try multiple backends if default fails
    if not cap.isOpened():
        print(f"✗ Default camera failed, trying DirectShow...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"✗ DirectShow failed, trying VFW...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_VFW)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {args.camera}")
        print("Available cameras: Try --camera 0, 1, 2, etc.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  SPACE = Capture and predict")
    print("  C     = Toggle continuous mode")
    print("  Q     = Quit")
    print("="*60 + "\n")
    
    continuous_mode = False
    last_prediction = None
    last_grayscale = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Preprocess
        frame_batch, grayscale = preprocess_frame(frame)
        last_grayscale = grayscale
        
        # Predict if continuous mode
        if continuous_mode:
            last_prediction = predict(model, frame_batch, class_names)
        
        # Create visualization
        display_frame = frame.copy()
        
        # Show grayscale preview (top-right corner)
        gray_resized = cv2.resize(grayscale, (200, 200))
        gray_3channel = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
        display_frame[10:210, -210:-10] = gray_3channel
        
        # Add info text
        y_offset = 40
        cv2.putText(display_frame, "BOTTLE CAP CLASSIFIER", (30, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if continuous_mode:
            cv2.putText(display_frame, "[CONTINUOUS MODE]", (30, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "[MANUAL MODE - Press SPACE]", (30, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Show prediction if available
        if last_prediction is not None:
            pred = last_prediction
            color = get_color(pred['class'])
            conf = pred['confidence']
            
            # Large prediction box
            text = f"{pred['class'].upper()}: {conf:.1%}"
            cv2.putText(display_frame, text, (30, y_offset + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Confidence bar
            bar_width = int(conf * 400)
            cv2.rectangle(display_frame, (30, y_offset + 150), (30 + bar_width, y_offset + 180),
                         color, -1)
            cv2.rectangle(display_frame, (30, y_offset + 150), (430, y_offset + 180),
                         (200, 200, 200), 2)
            cv2.putText(display_frame, f"Confidence: {conf:.1%}", (30, y_offset + 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show all probabilities
            y_start = y_offset + 250
            cv2.putText(display_frame, "All Predictions:", (30, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            for i, (class_name, prob) in enumerate(pred['probabilities'].items()):
                y_pos = y_start + 30 + (i * 25)
                prob_bar_width = int(prob * 300)
                
                # Class name
                cv2.putText(display_frame, f"{class_name:12}", (35, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Probability bar
                color_for_bar = get_color(class_name)
                cv2.rectangle(display_frame, (200, y_pos - 12), (200 + prob_bar_width, y_pos + 2),
                             color_for_bar, -1)
                
                # Percentage text
                cv2.putText(display_frame, f"{prob:.1%}", (510, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show instructions
        cv2.putText(display_frame, "SPACE=predict | C=continuous | Q=quit",
                   (30, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Display
        cv2.imshow('Cap Classifier - Camera Test', display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break
        
        elif key == ord('c') or key == ord('C'):
            continuous_mode = not continuous_mode
            mode_text = "ON (auto-predicting)" if continuous_mode else "OFF (manual)"
            print(f"\n>>> Continuous mode: {mode_text}\n")
        
        elif key == 32:  # SPACE
            if not continuous_mode:
                last_prediction = predict(model, frame_batch, class_names)
                pred = last_prediction
                print(f"\n>>> Prediction: {pred['class']} ({pred['confidence']:.1%})")
                for class_name, prob in pred['probabilities'].items():
                    print(f"    {class_name:12}: {prob:.1%}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()
