import tensorflow as tf
from data_loader import load_datasets
from model import build_model

DATA_DIR = "dataset"
EPOCHS = 15
MODEL_PATH = "models/bottle_cap_classifier.keras"

def main():
    print("📦 Loading datasets...")
    train_ds, val_ds, _ = load_datasets(DATA_DIR)

    print("🧠 Building model...")
    model = build_model(num_classes=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("🚀 Training started...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print(f"💾 Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
