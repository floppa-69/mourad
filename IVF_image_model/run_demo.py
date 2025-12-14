import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf


DEFAULT_INPUT_TENSOR = "input:0"
DEFAULT_OUTPUT_TENSOR = "probabilities:0"


def load_labels(labels_path: Path) -> List[str]:
    with labels_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    return sorted(files)


def preprocess_image(image_path: Path, target_size: int) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((target_size, target_size), resample=Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32)
        arr = arr / 127.5 - 1.0  # scale to [-1, 1] to match training
        return arr


def load_model(model_dir: Path):
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session(graph=tf.Graph())
    with sess.graph.as_default():
        tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], str(model_dir))
    return sess


def run_predictions(
    sess: tf.compat.v1.Session,
    images: List[Path],
    labels: List[str],
    input_tensor_name: str,
    output_tensor_name: str,
    target_size: int,
) -> List[Tuple[Path, int, float]]:
    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
    output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

    results = []
    for img_path in images:
        arr = preprocess_image(img_path, target_size)
        batch = np.expand_dims(arr, axis=0)
        probs = sess.run(output_tensor, feed_dict={input_tensor: batch})
        pred_idx = int(np.argmax(probs[0]))
        pred_prob = float(probs[0][pred_idx])
        results.append((img_path, pred_idx, pred_prob))
    return results


def write_predictions(output_path: Path, predictions: List[Tuple[Path, int, float]], labels: List[str]):
    with output_path.open("w", encoding="utf-8") as f:
        for img_path, pred_idx, pred_prob in predictions:
            label = labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx)
            f.write(f"{img_path} *** {pred_prob:.6f} *** {label}\n")


def main():
    parser = argparse.ArgumentParser(description="Run SavedModel predictions on a folder of images.")
    parser.add_argument("--model_dir", default="model", type=Path, help="Path to SavedModel directory.")
    parser.add_argument("--images_dir", default="examples", type=Path, help="Path to directory of images.")
    parser.add_argument("--labels", default="labels.txt", type=Path, help="Path to labels file.")
    parser.add_argument("--output", default="predictions.txt", type=Path, help="Where to write predictions.")
    parser.add_argument("--input_tensor", default=DEFAULT_INPUT_TENSOR, help="Input tensor name.")
    parser.add_argument("--output_tensor", default=DEFAULT_OUTPUT_TENSOR, help="Output tensor name.")
    parser.add_argument("--image_size", default=299, type=int, help="Model input size.")

    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    images_dir = args.images_dir.resolve()
    labels_path = args.labels.resolve()
    output_path = args.output.resolve()

    if not model_dir.exists():
        sys.exit(f"Model directory not found: {model_dir}")
    if not images_dir.exists():
        sys.exit(f"Images directory not found: {images_dir}")
    if not labels_path.exists():
        sys.exit(f"Labels file not found: {labels_path}")

    labels = load_labels(labels_path)
    images = list_images(images_dir)
    if not images:
        sys.exit(f"No images found in {images_dir}")

    sess = load_model(model_dir)
    predictions = run_predictions(
        sess,
        images,
        labels,
        args.input_tensor,
        args.output_tensor,
        args.image_size,
    )
    write_predictions(output_path, predictions, labels)
    print(f"Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()
