import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("PREGNANCY ANOMALY DETECTION - INFERENCE SCRIPT")
print("=" * 70)

# Load the trained model
print("\n[1] Loading trained model...")
try:
    with open("best_pregnancy_anomaly_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["features"]
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: Model file not found. Please train the model first.")
    exit(1)

# Load test data
print("\n[2] Loading test data...")
try:
    # Try to load data with labels first
    test_df = pd.read_csv("test_pregnancy_data.csv")
    has_labels = True
    print("✓ Test data loaded (with labels)")
except FileNotFoundError:
    # If not found, try blind data
    try:
        test_df = pd.read_csv("test_pregnancy_data_blind.csv")
        has_labels = False
        print("✓ Test data loaded (without labels)")
    except FileNotFoundError:
        print("❌ Error: No test data found. Please generate test data first.")
        exit(1)

print(f"  Total samples: {len(test_df)}")

# Extract features
print("\n[3] Preparing features...")
patient_ids = test_df["patient_id"].values

# Get feature columns (exclude patient_id and label columns if present)
feature_cols = [
    col
    for col in test_df.columns
    if col not in ["patient_id", "true_label", "true_condition"]
]

X_test = test_df[feature_cols]

# Verify all required features are present
missing_features = set(feature_names) - set(X_test.columns)
if missing_features:
    print(f"❌ Error: Missing features: {missing_features}")
    exit(1)

# Reorder columns to match training data
X_test = X_test[feature_names]
print(f"✓ Features prepared: {X_test.shape[1]} columns")

# Scale features
print("\n[4] Scaling features...")
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# Make predictions
print("\n[5] Making predictions...")
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# Create results dataframe
results_df = pd.DataFrame(
    {
        "patient_id": patient_ids,
        "predicted_label": predictions,
        "predicted_condition": [
            "Anomalous" if p == 1 else "Normal" for p in predictions
        ],
        "confidence_normal": probabilities[:, 0],
        "confidence_anomalous": probabilities[:, 1],
        "risk_score": probabilities[:, 1] * 100,  # Anomaly probability as percentage
    }
)

# Add true labels if available
if has_labels:
    results_df["true_label"] = test_df["true_label"].values
    results_df["true_condition"] = test_df["true_condition"].values
    results_df["correct_prediction"] = (
        results_df["predicted_label"] == results_df["true_label"]
    )

print("✓ Predictions complete")

# Display results summary
print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

prediction_counts = results_df["predicted_condition"].value_counts()
print(f"\nPredicted Normal: {prediction_counts.get('Normal', 0)}")
print(f"Predicted Anomalous: {prediction_counts.get('Anomalous', 0)}")

# Calculate statistics
avg_risk = results_df["risk_score"].mean()
high_risk_count = len(results_df[results_df["risk_score"] > 70])
medium_risk_count = len(
    results_df[(results_df["risk_score"] > 30) & (results_df["risk_score"] <= 70)]
)
low_risk_count = len(results_df[results_df["risk_score"] <= 30])

print(f"\nAverage Risk Score: {avg_risk:.2f}%")
print(f"\nRisk Distribution:")
print(f"  High Risk (>70%): {high_risk_count} patients")
print(f"  Medium Risk (30-70%): {medium_risk_count} patients")
print(f"  Low Risk (<30%): {low_risk_count} patients")

# If labels are available, show performance metrics
if has_labels:
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE ON TEST DATA")
    print("=" * 70)

    accuracy = (results_df["correct_prediction"].sum() / len(results_df)) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            results_df["true_label"],
            results_df["predicted_label"],
            target_names=["Normal", "Anomalous"],
        )
    )

    # Confusion matrix
    cm = confusion_matrix(results_df["true_label"], results_df["predicted_label"])
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")

    # ROC-AUC
    roc_auc = roc_auc_score(
        results_df["true_label"], results_df["confidence_anomalous"]
    )
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Display high-risk cases
print("\n" + "=" * 70)
print("HIGH-RISK CASES (Risk Score > 70%)")
print("=" * 70)

high_risk_cases = results_df[results_df["risk_score"] > 70].sort_values(
    "risk_score", ascending=False
)
if len(high_risk_cases) > 0:
    display_cols = ["patient_id", "predicted_condition", "risk_score"]
    if has_labels:
        display_cols.extend(["true_condition", "correct_prediction"])
    print(high_risk_cases[display_cols].to_string(index=False))
else:
    print("No high-risk cases detected.")

# Display detailed results for first 10 patients
print("\n" + "=" * 70)
print("DETAILED RESULTS - FIRST 10 PATIENTS")
print("=" * 70)

display_cols = ["patient_id", "predicted_condition", "risk_score"]
if has_labels:
    display_cols.extend(["true_condition", "correct_prediction"])

print(results_df[display_cols].head(10).to_string(index=False))

# Save results
results_filename = "prediction_results.csv"
results_df.to_csv(results_filename, index=False)
print(f"\n✓ Full results saved as '{results_filename}'")

# Generate visualization
print("\n[6] Generating visualizations...")

if has_labels:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 0],
        xticklabels=["Normal", "Anomalous"],
        yticklabels=["Normal", "Anomalous"],
    )
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    # 2. Risk Score Distribution
    axes[0, 1].hist(
        results_df["risk_score"],
        bins=20,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0, 1].axvline(x=30, color="green", linestyle="--", label="Low Risk Threshold")
    axes[0, 1].axvline(x=70, color="red", linestyle="--", label="High Risk Threshold")
    axes[0, 1].set_xlabel("Risk Score (%)")
    axes[0, 1].set_ylabel("Number of Patients")
    axes[0, 1].set_title("Risk Score Distribution")
    axes[0, 1].legend()

    # 3. Prediction Distribution by True Label
    pd.crosstab(results_df["true_condition"], results_df["predicted_condition"]).plot(
        kind="bar", ax=axes[1, 0]
    )
    axes[1, 0].set_title("Predictions by True Condition")
    axes[1, 0].set_xlabel("True Condition")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(title="Predicted")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Risk Score by True Label
    results_df.boxplot(column="risk_score", by="true_condition", ax=axes[1, 1])
    axes[1, 1].set_title("Risk Score Distribution by True Condition")
    axes[1, 1].set_xlabel("True Condition")
    axes[1, 1].set_ylabel("Risk Score (%)")
    plt.suptitle("")  # Remove default title

else:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Prediction Distribution
    prediction_counts.plot(kind="bar", ax=axes[0], color=["green", "red"])
    axes[0].set_title("Prediction Distribution")
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # 2. Risk Score Distribution
    axes[1].hist(
        results_df["risk_score"],
        bins=20,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    axes[1].axvline(x=30, color="green", linestyle="--", label="Low Risk Threshold")
    axes[1].axvline(x=70, color="red", linestyle="--", label="High Risk Threshold")
    axes[1].set_xlabel("Risk Score (%)")
    axes[1].set_ylabel("Number of Patients")
    axes[1].set_title("Risk Score Distribution")
    axes[1].legend()

plt.tight_layout()
plt.savefig("prediction_results.png", dpi=300, bbox_inches="tight")
print("✓ Visualizations saved as 'prediction_results.png'")

# Summary statistics
print("\n" + "=" * 70)
print("INFERENCE COMPLETE!")
print("=" * 70)
print(f"""
Files Generated:
✓ prediction_results.csv - Detailed predictions for all patients
✓ prediction_results.png - Visualization of results

Summary:
- Total Patients Analyzed: {len(results_df)}
- Predicted Normal: {prediction_counts.get("Normal", 0)}
- Predicted Anomalous: {prediction_counts.get("Anomalous", 0)}
- High Risk Cases: {high_risk_count}
- Average Risk Score: {avg_risk:.2f}%
""")

if has_labels:
    print(f"- Model Accuracy: {accuracy:.2f}%")
    print(f"- ROC-AUC Score: {roc_auc:.4f}")

print(
    "\nRecommendation: Review all high-risk cases (>70%) for further clinical evaluation."
)
print("=" * 70)
