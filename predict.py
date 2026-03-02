import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = (299, 299)


def main():
    parser = argparse.ArgumentParser(description="Predict the food category for an image.")
    parser.add_argument("--image", required=True, help="Path to a food image.")
    parser.add_argument("--model", default="artifacts/food101_inceptionv3.keras")
    parser.add_argument("--labels", default="artifacts/class_names.json")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not model_path.is_file() or not labels_path.is_file():
        raise SystemExit("Model/labels missing. Run train.py successfully first.")

    with open(labels_path, encoding="utf-8") as f:
        class_names = json.load(f)

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.inception_v3.preprocess_input(array)
    probabilities = tf.keras.models.load_model(model_path).predict(
        np.expand_dims(array, axis=0), verbose=0
    )[0]

    top_indices = np.argsort(probabilities)[::-1][:5]
    print("Top predictions:")
    for index in top_indices:
        print(f"{class_names[int(index)]}: {probabilities[index] * 100:.2f}%")


if __name__ == "__main__":
    main()
