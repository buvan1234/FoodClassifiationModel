import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

IMAGE_SIZE = (299, 299)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train an InceptionV3 Food-101 classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Optional limit for a quick test; omit to use all training images.")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Optional limit for a quick test; omit to use all test images.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def preprocess(example):
    image = tf.image.resize(example["image"], IMAGE_SIZE)
    image = tf.keras.applications.inception_v3.preprocess_input(
        tf.cast(image, tf.float32)
    )
    return image, example["label"]


def make_dataset(split, batch_size, limit=None, training=False):
    ds = tfds.load("food101", split=split, as_supervised=False, shuffle_files=training)
    if limit is not None:
        ds = ds.take(limit)
    if training:
        ds = ds.shuffle(min(limit or 5000, 5000), reshuffle_each_iteration=True)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(num_classes, learning_rate):
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="food_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="food_category")(x)
    model = models.Model(inputs, outputs, name="food101_inceptionv3")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading Food-101 dataset. The first run may download several GB.")
    info = tfds.builder("food101").info
    class_names = info.features["label"].names
    num_classes = len(class_names)
    if num_classes != 101:
        raise RuntimeError(f"Expected 101 classes, found {num_classes}.")

    train_ds = make_dataset("train", args.batch_size, args.max_train, training=True)
    test_ds = make_dataset("validation", args.batch_size, args.max_test, training=False)

    model = build_model(num_classes, args.learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "food101_inceptionv3.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Evaluating on the configured validation dataset...")
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    with open(args.output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "epochs_requested": args.epochs,
        "train_limit": args.max_train,
        "validation_limit": args.max_test,
        "note": "Accuracy reflects the exact dataset limits and configuration used for this run.",
    }
    with open(args.output_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and metadata under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
