import os
import joblib
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file, redirect, url_for

BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))


def _artifact_path(name):
    return os.path.join(BASE_DIR, name)


class IVFModelWrapper:
    def __init__(self, folder=BASE_DIR):
        self.folder = folder
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.metadata = None
        self._load_artifacts()

    def _artifact_path(self, name):
        return os.path.join(self.folder, name)

    def _load_artifacts(self):
        try:
            self.model = joblib.load(self._artifact_path("ivf_model.pkl"))
            self.scaler = joblib.load(self._artifact_path("scaler.pkl"))
            self.label_encoders = joblib.load(self._artifact_path("label_encoders.pkl"))
            self.feature_names = joblib.load(self._artifact_path("feature_names.pkl"))
            self.metadata = joblib.load(self._artifact_path("model_metadata.pkl"))
        except Exception as e:
            print("IVF model artifacts could not be loaded:", e)

    def is_ready(self):
        return self.model is not None and self.feature_names is not None and self.metadata is not None

    def preprocess_row(self, row: dict):
        X = pd.DataFrame([row])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan

        if self.label_encoders:
            for col, encoder in (self.label_encoders.items() if isinstance(self.label_encoders, dict) else []):
                if col in X.columns:
                    X[col] = X[col].astype(str).map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                    X[col] = encoder.transform(X[col])

        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        X = X[self.feature_names]

        if self.metadata and self.metadata.get("use_scaling") and self.scaler is not None:
            X_proc = self.scaler.transform(X)
        else:
            X_proc = X.values
        return X_proc

    def predict_single(self, row: dict):
        if not self.is_ready():
            raise RuntimeError("IVF model artifacts are not loaded")
        X_proc = self.preprocess_row(row)
        pred = int(self.model.predict(X_proc)[0])
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_proc)[:, 1][0])
        return pred, proba


class PregnancyModelWrapper:
    def __init__(self, folder=os.path.join(os.path.dirname(__file__), "../Pregnancy_anomaly_detection")):
        # folder may be absolute or relative; normalize
        self.folder = os.path.normpath(folder)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_artifacts()

    def _artifact_path(self, name):
        return os.path.join(self.folder, name)

    def _load_artifacts(self):
        try:
            p = self._artifact_path("best_pregnancy_anomaly_model.pkl")
            with open(p, "rb") as f:
                data = pickle.load(f)
            # Expecting dict with 'model','scaler','features'
            self.model = data.get("model")
            self.scaler = data.get("scaler")
            self.feature_names = data.get("features")
        except Exception as e:
            print("Pregnancy model artifacts could not be loaded:", e)

    def is_ready(self):
        return self.model is not None and self.scaler is not None and self.feature_names is not None

    def predict_batch(self):
        # Read test csv from its folder
        test_path = self._artifact_path("test_pregnancy_data.csv")
        blind_path = self._artifact_path("test_pregnancy_data_blind.csv")
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            has_labels = True
        elif os.path.exists(blind_path):
            df = pd.read_csv(blind_path)
            has_labels = False
        else:
            raise FileNotFoundError("No pregnancy test data found")

        patient_ids = df["patient_id"].values
        feature_cols = [c for c in df.columns if c not in ["patient_id", "true_label", "true_condition"]]
        X = df[feature_cols]
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)

        results = pd.DataFrame({
            "patient_id": patient_ids,
            "predicted_label": preds,
            "predicted_condition": ["Anomalous" if p == 1 else "Normal" for p in preds],
            "confidence_normal": probas[:, 0],
            "confidence_anomalous": probas[:, 1],
            "risk_score": probas[:, 1] * 100,
        })

        if has_labels:
            results["true_label"] = df["true_label"].values
            results["true_condition"] = df["true_condition"].values
            results["correct_prediction"] = results["predicted_label"] == results["true_label"]

        out = self._artifact_path("prediction_results.csv")
        results.to_csv(out, index=False)
        return results, out


IVF = IVFModelWrapper()
PREG = PregnancyModelWrapper(folder=os.path.join(BASE_DIR, "../Pregnancy_anomaly_detection"))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ivf_loaded": IVF.is_ready(),
        "pregnancy_loaded": PREG.is_ready(),
    })


@app.route("/ivf", methods=["GET"])
def ivf_index():
    # Show IVF model status and provide links
    ivf_ready = IVF.is_ready()
    # preview predictions CSV if exists
    preds_path = _artifact_path("test_predictions.csv")
    preview_html = None
    if os.path.exists(preds_path):
        try:
            df = pd.read_csv(preds_path)
            preview_html = df.head(10).to_html(classes="table table-sm", index=False)
        except Exception:
            preview_html = None
    csv_path = preds_path if os.path.exists(preds_path) else None
    return render_template("ivf.html", ivf_ready=ivf_ready, preds_preview=preview_html, csv_path=csv_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", ivf_loaded=IVF.is_ready(), pregnancy_loaded=PREG.is_ready())


@app.route("/ivf/predict", methods=["POST"])
def ivf_predict():
    if not IVF.is_ready():
        return render_template("error.html", message="IVF model artifacts not loaded"), 500

    payload = request.get_json(silent=True) or request.form.to_dict()
    features = payload.get("features") if isinstance(payload.get("features"), dict) else payload
    if not features:
        return render_template("error.html", message="No features provided"), 400

    # cast any empty strings to None
    features = {k: (None if v == "" else v) for k, v in features.items()}
    try:
        pred, proba = IVF.predict_single(features)
    except Exception as e:
        return render_template("error.html", message=f"Prediction failed: {e}"), 500

    return render_template("ivf_result.html", predicted_class=pred, probability=proba)


@app.route("/ivf/run_batch", methods=["POST", "GET"])
def ivf_run_batch():
    if not IVF.is_ready():
        return render_template("error.html", message="IVF model artifacts not loaded"), 500

    test_path = _artifact_path("ivf_test_dataset.csv")
    if not os.path.exists(test_path):
        return render_template("error.html", message="Test dataset not found (ivf_test_dataset.csv)"), 400

    df = pd.read_csv(test_path)
    target_col = IVF.metadata.get("target_column") if IVF.metadata else None
    exclude_cols = ["patient_id", "embryo_health_score", "embryo_quality_class"]

    if target_col is None or target_col not in df.columns:
        return render_template("error.html", message="Target column missing in metadata or dataset"), 400

    X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    y = df[target_col]

    if IVF.label_encoders:
        for col, encoder in (IVF.label_encoders.items() if isinstance(IVF.label_encoders, dict) else []):
            if col in X.columns:
                X[col] = X[col].astype(str).map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                X[col] = encoder.transform(X[col])

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    if IVF.feature_names:
        X = X[IVF.feature_names]

    if IVF.metadata and IVF.metadata.get("use_scaling") and IVF.scaler is not None:
        X_proc = IVF.scaler.transform(X)
    else:
        X_proc = X.values

    preds = IVF.model.predict(X_proc)
    probas = IVF.model.predict_proba(X_proc)[:, 1] if hasattr(IVF.model, "predict_proba") else [None] * len(preds)

    results = df.copy()
    results["predicted_class"] = preds
    results["prediction_probability"] = probas
    results["correct_prediction"] = (results["predicted_class"] == y).astype(int)

    out = _artifact_path("test_predictions.csv")
    results.to_csv(out, index=False)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = float(accuracy_score(y, preds))
    precision = float(precision_score(y, preds))
    recall = float(recall_score(y, preds))
    f1 = float(f1_score(y, preds))
    try:
        roc_auc = float(roc_auc_score(y, probas))
    except Exception:
        roc_auc = None

    preview_html = results.head(10).to_html(classes="table table-sm", index=False)
    return render_template("ivf_batch.html", n_samples=len(results), accuracy=accuracy, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc, preview_html=preview_html, csv_path=out)


@app.route('/pregnancy', methods=['GET'])
def pregnancy_index():
    preg_ready = PREG.is_ready()
    # preview pregnancy results if exists
    preds_path = os.path.join(os.path.dirname(PREG._artifact_path('')), 'prediction_results.csv') if PREG.is_ready() else None
    preview_html = None
    if preds_path and os.path.exists(preds_path):
        try:
            df = pd.read_csv(preds_path)
            preview_html = df.head(10).to_html(classes='table table-sm', index=False)
        except Exception:
            preview_html = None
    return render_template('pregnancy.html', preg_ready=preg_ready, preds_preview=preview_html)


@app.route('/pregnancy/run_batch', methods=['POST', 'GET'])
def pregnancy_run_batch():
    if not PREG.is_ready():
        return render_template('error.html', message='Pregnancy model artifacts not loaded'), 500
    try:
        results, out = PREG.predict_batch()
    except Exception as e:
        return render_template('error.html', message=f'Failed to run pregnancy batch: {e}'), 500

    preview_html = results.head(10).to_html(classes='table table-sm', index=False)
    return render_template('pregnancy_batch.html', n_samples=len(results), preview_html=preview_html, csv_path=out)


@app.route('/download')
def download_file():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return render_template('error.html', message='File not found'), 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    # Run locally for development (debug disabled to avoid reloader in background)
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting server on 127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
