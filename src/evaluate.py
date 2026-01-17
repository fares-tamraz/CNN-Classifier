import tensorflow as tf
from data_loader import load_datasets

DATA_DIR = "dataset"
MODEL_PATH = "models/bottle_cap_classifier.keras"

def main():
    print("📦 Loading test dataset...")
    _, _, test_ds = load_datasets(DATA_DIR)

    print("🧠 Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("📊 Evaluating model...")
    loss, accuracy = model.evaluate(test_ds)

    print(f"✅ Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
