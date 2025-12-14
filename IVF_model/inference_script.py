import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("IVF EMBRYO HEALTH PREDICTION - MODEL TESTING")
print("=" * 80)

# Load test data
print("\n[1] Loading test data...")
try:
    test_df = pd.read_csv("ivf_test_dataset.csv")
    print(f"✓ Loaded {len(test_df)} test samples")
except FileNotFoundError:
    print("✗ Error: ivf_test_dataset.csv not found!")
    print("  Please run the test data generation script first.")
    exit(1)

# Load model and preprocessing objects
print("\n[2] Loading trained model and preprocessing objects...")
try:
    model = joblib.load("ivf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    metadata = joblib.load("model_metadata.pkl")
    print(f"✓ Model loaded: {metadata['model_type']}")
    print(f"✓ Training accuracy: {metadata['accuracy']:.4f}")
    print(f"✓ Training ROC-AUC: {metadata['roc_auc']:.4f}")
except FileNotFoundError as e:
    print(f"✗ Error: Required files not found!")
    print("  Please run the training script first to generate:")
    print("  - ivf_model.pkl")
    print("  - scaler.pkl")
    print("  - label_encoders.pkl")
    print("  - feature_names.pkl")
    print("  - model_metadata.pkl")
    exit(1)

# Prepare test features
print("\n[3] Preparing test features...")
target_col = metadata["target_column"]
exclude_cols = ["patient_id", "embryo_health_score", "embryo_quality_class"]

X_test = test_df.drop(columns=exclude_cols)
y_test = test_df[target_col]

print(f"✓ Test features: {X_test.shape[1]} columns")
print(f"✓ Test samples: {len(X_test)}")

# Encode categorical variables
print("\n[4] Encoding categorical variables...")
for col, encoder in label_encoders.items():
    if col in X_test.columns:
        # Handle unseen categories
        X_test[col] = (
            X_test[col]
            .astype(str)
            .map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
        )
        X_test[col] = encoder.transform(X_test[col])
        print(f"✓ Encoded: {col}")

# Handle missing values
X_test = X_test.fillna(X_test.median())

# Ensure feature order matches training
X_test = X_test[feature_names]

# Scale if needed
if metadata["use_scaling"]:
    print("\n[5] Scaling features...")
    X_test_processed = scaler.transform(X_test)
    print("✓ Features scaled")
else:
    print("\n[5] No scaling needed for this model")
    X_test_processed = X_test

# Make predictions
print("\n[6] Making predictions...")
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
print(f"✓ Generated predictions for {len(y_pred)} samples")

# Calculate metrics
print("\n" + "=" * 80)
print("TEST SET PERFORMANCE METRICS")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Detailed classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(
    classification_report(
        y_test, y_pred, target_names=["Poor Quality (0)", "Good Quality (1)"], digits=4
    )
)

# Confusion matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
cm = confusion_matrix(y_test, y_pred)
print("\n                 Predicted")
print("               0         1")
print(f"Actual 0    {cm[0, 0]:4d}     {cm[0, 1]:4d}")
print(f"       1    {cm[1, 0]:4d}     {cm[1, 1]:4d}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn} (correctly predicted poor quality)")
print(f"False Positives: {fp} (incorrectly predicted good quality)")
print(f"False Negatives: {fn} (incorrectly predicted poor quality)")
print(f"True Positives:  {tp} (correctly predicted good quality)")

# Calculate additional metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Positive Predictive Value (PPV):  {ppv:.4f}")
print(f"Negative Predictive Value (NPV):  {npv:.4f}")

# Save predictions
print("\n" + "=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

results_df = test_df.copy()
results_df["predicted_class"] = y_pred
results_df["prediction_probability"] = y_pred_proba
results_df["correct_prediction"] = (y_pred == y_test).astype(int)

output_file = "test_predictions.csv"
results_df.to_csv(output_file, index=False)
print(f"✓ Predictions saved to: {output_file}")

# Show sample predictions
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (First 10 cases)")
print("=" * 80)
sample_cols = [
    "patient_id",
    "age",
    "amh_level",
    "reached_blastocyst",
    "embryo_quality_class",
    "predicted_class",
    "prediction_probability",
    "correct_prediction",
]
print(results_df[sample_cols].head(10).to_string(index=False))

# Analysis of errors
print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

false_positives = results_df[
    (results_df["embryo_quality_class"] == 0) & (results_df["predicted_class"] == 1)
]
false_negatives = results_df[
    (results_df["embryo_quality_class"] == 1) & (results_df["predicted_class"] == 0)
]

print(f"\nFalse Positives: {len(false_positives)}")
if len(false_positives) > 0:
    print("Average characteristics:")
    print(f"  Age: {false_positives['age'].mean():.1f}")
    print(f"  AMH: {false_positives['amh_level'].mean():.2f}")
    print(f"  Health Score: {false_positives['embryo_health_score'].mean():.1f}")

print(f"\nFalse Negatives: {len(false_negatives)}")
if len(false_negatives) > 0:
    print("Average characteristics:")
    print(f"  Age: {false_negatives['age'].mean():.1f}")
    print(f"  AMH: {false_negatives['amh_level'].mean():.2f}")
    print(f"  Health Score: {false_negatives['embryo_health_score'].mean():.1f}")

# Prediction confidence analysis
print("\n" + "=" * 80)
print("PREDICTION CONFIDENCE ANALYSIS")
print("=" * 80)

high_confidence = results_df[
    (
        (results_df["prediction_probability"] > 0.8)
        | (results_df["prediction_probability"] < 0.2)
    )
]
low_confidence = results_df[
    (
        (results_df["prediction_probability"] >= 0.4)
        & (results_df["prediction_probability"] <= 0.6)
    )
]

print(f"\nHigh confidence predictions (p < 0.2 or p > 0.8): {len(high_confidence)}")
print(f"  Accuracy: {high_confidence['correct_prediction'].mean():.4f}")

print(f"\nLow confidence predictions (0.4 <= p <= 0.6): {len(low_confidence)}")
print(f"  Accuracy: {low_confidence['correct_prediction'].mean():.4f}")

# Clinical implications
print("\n" + "=" * 80)
print("CLINICAL IMPLICATIONS")
print("=" * 80)

print(f"""
Model Performance Summary:
- Overall Accuracy: {accuracy:.1%}
- Successfully identified {tp} good quality embryos
- Correctly rejected {tn} poor quality embryos
- Missed {fn} good quality embryos (could have been implanted)
- Falsely identified {fp} poor quality embryos as good (unnecessary procedures)

Clinical Considerations:
1. The model shows {"high" if roc_auc > 0.85 else "moderate" if roc_auc > 0.75 else "acceptable"} discriminative ability (ROC-AUC: {roc_auc:.3f})
2. Sensitivity (recall): {sensitivity:.1%} - proportion of good embryos correctly identified
3. Specificity: {specificity:.1%} - proportion of poor embryos correctly rejected
4. False negative rate: {fn / (tp + fn):.1%} - risk of missing viable embryos
5. False positive rate: {fp / (fp + tn):.1%} - risk of selecting non-viable embryos
""")

# Model comparison with training
print("\n" + "=" * 80)
print("TRAINING vs TEST PERFORMANCE")
print("=" * 80)

print(f"{'Metric':<20} {'Training':<15} {'Test':<15} {'Difference'}")
print("-" * 80)
print(
    f"{'Accuracy':<20} {metadata['accuracy']:<15.4f} {accuracy:<15.4f} {accuracy - metadata['accuracy']:+.4f}"
)
print(
    f"{'ROC-AUC':<20} {metadata['roc_auc']:<15.4f} {roc_auc:<15.4f} {roc_auc - metadata['roc_auc']:+.4f}"
)

if abs(accuracy - metadata["accuracy"]) > 0.05:
    print("\n⚠ Warning: Significant performance drop detected!")
    print("  Consider retraining with more diverse data or adjusting model parameters.")
elif accuracy > metadata["accuracy"]:
    print("\n✓ Model generalizes well to test data!")
else:
    print("\n✓ Performance is consistent with training results.")

print("\n" + "=" * 80)
print("TESTING COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print(f"  - {output_file}")
print("\nModel is ready for clinical validation and deployment.")
print("=" * 80)
