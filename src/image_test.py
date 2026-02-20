"""
Image-based test for bottle cap classifier (when camera doesn't work).

Test with static images from the dataset or your own test images.

Usage:
  python src/image_test.py --image path/to/image.jpg
  python src/image_test.py --image path/to/image.jpg --confidence_threshold 0.5
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


def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> tuple:
    """
    Preprocess image same way as training:
    1. Load image
    2. Convert to grayscale
    3. Resize to 224x224
    4. Normalize to 0-1
    5. Add batch dimension
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, target_size)
    
    # Normalize to 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    # Add channel dimension (224, 224) -> (224, 224, 1)
    with_channel = np.expand_dims(normalized, axis=-1)
    
    # Add batch dimension (224, 224, 1) -> (1, 224, 224, 1)
    batch = np.expand_dims(with_channel, axis=0)
    
    return batch, resized, original


def predict(model: tf.keras.Model, image_batch: np.ndarray, class_names: list) -> Dict:
    """Run prediction on preprocessed image."""
    predictions = model.predict(image_batch, verbose=0)
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
    parser = argparse.ArgumentParser(description='Test cap classifier on image files')
    parser.add_argument('--model', type=str, default='models/cap_classifier_best.keras',
                        help='Path to trained model')
    parser.add_argument('--class_names', type=str, default='models/class_names.json',
                        help='Path to class names JSON')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file to classify')
    
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
    print(f"✓ Classes: {class_names}\n")
    
    # Load and process image
    print(f"Loading image: {args.image}")
    try:
        image_batch, grayscale, original = preprocess_image(args.image)
        print("✓ Image loaded and preprocessed")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return
    
    # Make prediction
    print("\nRunning prediction...")
    prediction = predict(model, image_batch, class_names)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {prediction['class'].upper()}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print("\nAll Class Probabilities:")
    print("-" * 60)
    
    # Sort by probability (highest first)
    sorted_probs = sorted(prediction['probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs:
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{class_name:12} | {bar} | {prob:6.2%}")
    
    print("="*60)
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Resize original for display
    display_width = 600
    aspect_ratio = original.shape[0] / original.shape[1]
    display_height = int(display_width * aspect_ratio)
    display_img = cv2.resize(original, (display_width, display_height))
    
    # Add side panel with predictions
    panel_width = 400
    panel_height = display_height
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 40
    
    # Add text to panel
    y_offset = 30
    cv2.putText(panel, "PREDICTION", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Main prediction (large)
    pred_color = get_color(prediction['class'])
    y_offset += 80
    cv2.putText(panel, prediction['class'].upper(), (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
    
    # Confidence
    y_offset += 50
    conf_text = f"{prediction['confidence']:.1%}"
    cv2.putText(panel, f"Confidence:", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(panel, conf_text, (20, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Confidence bar
    bar_width = int(prediction['confidence'] * 360)
    cv2.rectangle(panel, (20, y_offset + 50), (20 + bar_width, y_offset + 70),
                 pred_color, -1)
    cv2.rectangle(panel, (20, y_offset + 50), (380, y_offset + 70),
                 (150, 150, 150), 2)
    
    # All predictions
    y_offset += 120
    cv2.putText(panel, "All Classes:", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    for i, (class_name, prob) in enumerate(sorted_probs):
        y_pos = y_offset + 30 + (i * 35)
        
        # Class name
        cv2.putText(panel, class_name, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Probability bar
        bar_w = int(prob * 280)
        color_for_bar = get_color(class_name)
        cv2.rectangle(panel, (100, y_pos - 12), (100 + bar_w, y_pos + 2),
                     color_for_bar, -1)
        
        # Percentage
        cv2.putText(panel, f"{prob:.0%}", (330, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine images
    result = np.hstack([display_img, panel])
    
    # Display
    cv2.imshow('Bottle Cap Classifier - Image Test', result)
    print("✓ Visualization displayed (press any key to close)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
