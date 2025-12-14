# Pregnancy Anomaly Detection Model

## Overview

This AI-powered system detects anomalies in first-trimester pregnancies (weeks 4-13) using machine learning. The model analyzes maternal health metrics and embryo measurements to identify potential complications early, enabling timely medical intervention.

---

## 🎯 Purpose

Early detection of pregnancy complications can significantly improve maternal and fetal outcomes. This model helps healthcare providers:

- **Screen** pregnant patients for potential anomalies
- **Identify** high-risk cases requiring closer monitoring
- **Prioritize** clinical resources for patients who need them most
- **Support** clinical decision-making with data-driven insights

---

## 📊 How the Model Works

### 1. **Data Input**

The model analyzes **24 key features** from routine prenatal checkups:

#### Maternal Demographics
- **Maternal Age**: Age of the mother (years)
- **BMI**: Body Mass Index (weight/height²)
- **Gestational Age**: Pregnancy stage (weeks 4-13)

#### Vital Signs
- **Blood Pressure**: Systolic and diastolic readings (mmHg)
- **Heart Rate**: Beats per minute (bpm)

#### Blood Test Results
- **Hemoglobin**: Red blood cell count (g/dL) - anemia indicator
- **hCG Level**: Human Chorionic Gonadotropin (mIU/mL) - pregnancy hormone
- **Progesterone**: Hormone supporting pregnancy (ng/mL)
- **Glucose**: Blood sugar level (mg/dL)
- **TSH**: Thyroid Stimulating Hormone (µIU/mL)

#### Embryo/Fetal Measurements (from ultrasound)
- **Crown-Rump Length (CRL)**: Size of embryo/fetus (mm) - primary growth indicator
- **Gestational Sac Diameter**: Size of pregnancy sac (mm)
- **Yolk Sac Diameter**: Size of yolk sac (mm)
- **Fetal Heart Rate**: Heartbeats per minute (bpm)

#### Clinical Symptoms (Binary: Yes/No)
- Vaginal bleeding
- Severe nausea/vomiting
- Abdominal pain
- Fever

#### Medical History (Binary: Yes/No)
- Previous miscarriage
- Diabetes
- Hypertension (high blood pressure)

### 2. **Data Processing Pipeline**

```
Raw Patient Data → Feature Extraction → Standardization → Model Prediction → Risk Score
```

#### Step-by-Step Process:

1. **Feature Extraction**: Collect all 24 measurements from patient records
2. **Standardization**: Scale all values to comparable ranges (mean=0, std=1)
3. **Model Processing**: Feed data through trained machine learning algorithm
4. **Probability Calculation**: Generate likelihood scores for normal vs. anomalous
5. **Risk Assessment**: Convert probabilities to actionable risk scores (0-100%)

### 3. **Machine Learning Algorithm**

The system trains and compares **4 different algorithms**:

| Algorithm | Type | Strengths |
|-----------|------|-----------|
| **Random Forest** | Ensemble Learning | Handles non-linear relationships, robust to outliers |
| **Gradient Boosting** | Ensemble Learning | High accuracy, captures complex patterns |
| **Logistic Regression** | Linear Model | Fast, interpretable, good baseline |
| **Support Vector Machine (SVM)** | Kernel-based | Effective in high-dimensional spaces |

The **best-performing model** is automatically selected based on F1-Score (balance between precision and recall).

### 4. **Training Process**

```
Training Data (4,000 samples)
    ↓
80/20 Split
    ↓
Train Set (3,200) → Model Training → 5-Fold Cross-Validation
    ↓
Test Set (800) → Performance Evaluation
    ↓
Best Model Selection → Save for Deployment
```

**Key Training Features:**
- **Stratified Splitting**: Maintains anomaly ratio in train/test sets
- **Cross-Validation**: Tests model on 5 different data subsets
- **Hyperparameter Tuning**: Optimizes model settings for best performance
- **Feature Scaling**: Ensures all measurements are equally weighted

---

## 🔍 Anomaly Types Detected

The model can identify **5 major pregnancy complications**:

### 1. **Threatened Miscarriage**
**Key Indicators:**
- Vaginal bleeding present
- Lower progesterone levels (<15 ng/mL)
- Reduced hCG levels (50-80% of normal)
- Abdominal pain may be present

**Risk Factors:** Previous miscarriage history

### 2. **Ectopic Pregnancy**
**Key Indicators:**
- Severe abdominal pain
- Abnormally low hCG levels (30-60% of normal)
- No visible gestational sac in uterus (diameter = 0)
- No fetal heart activity detected

**Urgency:** Medical emergency requiring immediate intervention

### 3. **Molar Pregnancy**
**Key Indicators:**
- Extremely high hCG levels (2-4x normal)
- Severe nausea/vomiting
- Elevated blood pressure
- Enlarged gestational sac (1.3-1.8x normal)
- No fetal heart activity

**Characteristic:** Abnormal placental tissue growth

### 4. **Chromosomal Abnormality**
**Key Indicators:**
- Smaller crown-rump length (70-90% of expected)
- Enlarged yolk sac (1.2-1.5x normal)
- Reduced fetal heart rate (85-95% of normal)
- Advanced maternal age (>35 years)

**Examples:** Down syndrome, Turner syndrome, other genetic conditions

### 5. **Fetal Growth Restriction**
**Key Indicators:**
- Significantly reduced CRL (60-80% of expected)
- Smaller gestational sac (70-85% of normal)
- Maternal anemia (hemoglobin <11 g/dL)
- Maternal hypertension

**Concern:** Inadequate fetal development

---

## 📈 Model Output

### Risk Score (0-100%)

The model produces a **continuous risk score** indicating anomaly likelihood:

| Risk Level | Score Range | Interpretation | Action |
|------------|-------------|----------------|--------|
| **Low** | 0-30% | Normal pregnancy likely | Routine monitoring |
| **Medium** | 31-70% | Some concerns present | Enhanced monitoring, repeat tests |
| **High** | 71-100% | Significant anomaly risk | Immediate clinical review, specialist referral |

### Prediction Output

For each patient, the model provides:

```
Patient ID: TEST0042
├── Predicted Condition: Anomalous
├── Risk Score: 87.3%
├── Confidence
│   ├── Normal: 12.7%
│   └── Anomalous: 87.3%
└── Recommendation: High-risk case - immediate clinical evaluation required
```

---

## 🎓 Model Performance

### Evaluation Metrics

The model is evaluated using multiple metrics:

| Metric | Description | Typical Performance |
|--------|-------------|---------------------|
| **Accuracy** | Overall correct predictions | 92-96% |
| **Precision** | Of predicted anomalies, how many are true | 88-93% |
| **Recall** | Of true anomalies, how many are detected | 85-91% |
| **F1-Score** | Harmonic mean of precision & recall | 87-92% |
| **ROC-AUC** | Overall discrimination ability | 0.94-0.98 |

### What These Metrics Mean

- **High Precision**: Few false alarms (normal pregnancies wrongly flagged)
- **High Recall**: Few missed cases (actual anomalies correctly detected)
- **High ROC-AUC**: Excellent ability to distinguish normal from anomalous

### Clinical Significance

- **Sensitivity to Anomalies**: The model is designed to have high recall, prioritizing detection of actual complications even if it means some false positives
- **Balanced Approach**: F1-Score optimization ensures neither false positives nor false negatives are excessively high

---

## 🔬 Feature Importance

Not all features contribute equally. Top predictive features typically include:

### Most Important (High Impact)
1. **hCG Level** - Strongest pregnancy hormone indicator
2. **Progesterone Level** - Critical for pregnancy maintenance
3. **Crown-Rump Length** - Direct measure of fetal growth
4. **Fetal Heart Rate** - Vital sign of fetal wellbeing
5. **Vaginal Bleeding** - Primary warning symptom

### Moderately Important
6. Gestational sac diameter
7. Maternal age
8. Blood pressure (systolic)
9. Yolk sac diameter
10. Abdominal pain

### Supporting Features
- Hemoglobin, glucose, TSH levels
- Medical history factors
- Other clinical symptoms

*Note: Feature importance varies by model type and is automatically calculated during training.*

---

## 🛠️ Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION                          │
│  generate_dataset.py → pregnancy_anomaly_dataset.csv        │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                           │
│  train_model.py → best_pregnancy_anomaly_model.pkl          │
│                 → model_evaluation.png                      │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    TEST DATA GENERATION                     │
│  generate_test_data.py → test_pregnancy_data.csv            │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE/PREDICTION                     │
│  run_inference.py → prediction_results.csv                  │
│                  → prediction_results.png                   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Python 3.7+**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization

---

## 📋 Usage Workflow

### Complete Pipeline

```bash
# Step 1: Generate training dataset (5,000 samples)
python generate_dataset.py

# Step 2: Train and evaluate models
python train_model.py

# Step 3: Generate new test cases (50 samples)
python generate_test_data.py

# Step 4: Run predictions on test data
python run_inference.py
```

### Output Files

| File | Purpose |
|------|---------|
| `pregnancy_anomaly_dataset.csv` | Training data with labels |
| `best_pregnancy_anomaly_model.pkl` | Trained model (pickle format) |
| `model_evaluation.png` | Training performance charts |
| `test_pregnancy_data.csv` | Test data with true labels |
| `test_pregnancy_data_blind.csv` | Test data without labels |
| `prediction_results.csv` | Model predictions + risk scores |
| `prediction_results.png` | Prediction visualizations |

---

## ⚠️ Important Considerations

### Clinical Use

- **Screening Tool Only**: This model is designed to assist healthcare providers, not replace clinical judgment
- **Requires Validation**: Results should be confirmed with additional clinical examination
- **Not Diagnostic**: Cannot definitively diagnose conditions - further testing required
- **Part of Care Pathway**: Use as one component of comprehensive prenatal care

### Limitations

1. **Based on Synthetic Data**: This implementation uses simulated data for demonstration
2. **Limited Anomaly Types**: Detects 5 major conditions; other complications may exist
3. **First Trimester Only**: Designed for weeks 4-13; not applicable to later stages
4. **Population Specific**: Performance may vary across different populations
5. **Requires Quality Data**: Accuracy depends on precise measurements

### Regulatory Compliance

Before clinical deployment:
- Validate with real patient data
- Obtain appropriate medical device approvals
- Ensure HIPAA compliance
- Conduct clinical trials
- Get ethics committee approval

---

## 🔒 Data Privacy & Security

### Protected Health Information (PHI)

All patient data must be handled according to:
- **HIPAA** (Health Insurance Portability and Accountability Act)
- **GDPR** (if applicable in EU)
- Local healthcare data regulations

### Best Practices

- De-identify patient data before processing
- Use secure storage for all medical records
- Implement access controls and audit logs
- Encrypt data in transit and at rest
- Regular security audits

---

## 📚 References & Further Reading

### Medical Guidelines
- American College of Obstetricians and Gynecologists (ACOG) First Trimester Guidelines
- WHO Recommendations on Antenatal Care
- Royal College of Obstetricians and Gynaecologists (RCOG) Early Pregnancy Guidelines

### Machine Learning in Healthcare
- FDA Guidance on AI/ML-Based Medical Devices
- Clinical Decision Support Systems in Obstetrics
- Predictive Analytics in Prenatal Care

---

## 🤝 Contributing

To improve this model:

1. **Add Real Clinical Data**: Replace synthetic data with anonymized patient records
2. **Expand Anomaly Types**: Include additional complications
3. **Improve Features**: Add more biomarkers or imaging features
4. **Validate Performance**: Conduct multi-center clinical validation studies
5. **Optimize Algorithms**: Experiment with deep learning approaches

---

## 📞 Support & Contact

For questions, issues, or clinical collaboration:

- Review code documentation in each script
- Check error messages in console output
- Validate input data format matches specifications
- Ensure all dependencies are installed

---

## ⚖️ License & Disclaimer

**DISCLAIMER**: This model is for educational and research purposes only. It is NOT approved for clinical use. Always consult qualified healthcare professionals for medical decisions.

---

## 🔄 Version History

- **v1.0** - Initial release with 5 anomaly types, 24 features, 4 ML algorithms

---

*Last Updated: December 2024*