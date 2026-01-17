import tensorflow as tf
import numpy as np
import sys

IMAGE_SIZE = (224, 224)
MODEL_PATH = "models/bottle_cap_classifier.keras"
CLASS_NAMES = ["cap_faulty", "cap_ok"]

def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    confidence = tf.nn.softmax(predictions[0])

    class_idx = np.argmax(confidence)
    label = CLASS_NAMES[class_idx]
    score = float(confidence[class_idx])

    print(f"🔍 Prediction: {label}")
    print(f"📈 Confidence: {score:.2%}")

    # Factory-style reject logic
    if score < 0.90:
        print("⚠️ Low confidence — send to manual inspection")
    elif label == "cap_faulty":
        print("❌ REJECT ITEM")
    else:
        print("✅ ACCEPT ITEM")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
