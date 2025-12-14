# Standalone IVF Classifier Demo

This folder contains a ready-to-run demo for the embryo quality classifier (Inception V3) exported as a TensorFlow SavedModel. It uses a handful of sample images and produces predictions to a text file.

## Contents
- `model/` — SavedModel (`saved_model.pb` + `variables/`)
- `examples/` — sample embryo images (copied from the project test set)
- `labels.txt` — class names used by the model
- `run_demo.py` — simple prediction script
- `requirements.txt` — minimal dependencies

## Prerequisites
- Python 3.7 (matches the training environment)
- (Recommended) a virtual environment

## Setup (Windows PowerShell)
```powershell
# From repo root
python -m venv standalone\.venv
standalone\.venv\Scripts\activate
pip install -r standalone\requirements.txt
```

## Run predictions
```powershell
# From repo root, with the venv activated
python standalone\run_demo.py --model_dir standalone\model --images_dir standalone\examples --labels standalone\labels.txt --output standalone\predictions.txt
```

Output file format (one line per image):
```
<image_path> *** <probability> *** <predicted_label>
```

## Custom inputs
Place your own JPEG/PNG/BMP files into `standalone/examples/` (or another folder) and point `--images_dir` to it. Ensure images are RGB; the script will resize to 299x299 and scale to [-1, 1] to match training.

## Tensor names (if needed)
- Input: `input:0`
- Output: `probabilities:0`

## Troubleshooting
- If you see "No images found", confirm the `--images_dir` path and that it contains image files.
- If TensorFlow errors about missing DLLs on Windows, install the Visual C++ Redistributable and ensure you are using Python 3.7.
