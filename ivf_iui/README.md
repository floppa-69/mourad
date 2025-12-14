# Fertility Models - Inference Deployment Package

Complete standalone package for making predictions with trained IUI and IVF fertility models using PyTorch neural networks.

## 📁 Folder Structure

```
inference_deployment/
├── iui_inference.py              # IUI model inference script
├── ivf_inference.py              # IVF model inference script
├── examples/
│   ├── iui_example_patients.csv  # 4 example IUI patient records
│   ├── ivf_example_patients.csv  # 4 example IVF patient records
│   └── (place your data here)
└── README.md                     # This file
```

This folder should be placed in the same parent directory as your `models/` folder containing the trained models.

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn
```

Or with conda:
```bash
conda install pytorch pandas numpy scikit-learn
```

### Run IUI Inference

```bash
python iui_inference.py
```

**Output:**
```
Using device: cpu

Loading model from: ../models/model_m_iui_pytorch.pth
Loading training data from: ../models/iui_synthetic_dataset.csv

============================================================
IUI MODEL PREDICTIONS
============================================================

--- Patient 1 ---
Age: 28, Partner Age: 30, BMI: 24.5
PCOS: No, Cycle: Regular, AMH: 2.5, FSH: 5.0
...

Prediction: POSITIVE (Pregnancy likely)
Confidence: 78.45%
Probabilities - Negative: 21.5%, Positive: 78.5%
```

### Run IVF Inference

```bash
python ivf_inference.py
```

---

## 📊 Using Your Own Patient Data

### Option 1: Load from CSV File

```python
# In iui_inference.py or ivf_inference.py, replace the sample_patients definition with:

df_patients = pd.read_csv('examples/your_patients.csv')
sample_patients = []
for idx, row in df_patients.iterrows():
    sample_patients.append(row.to_dict())
```

### Option 2: Add Single Patient

```python
new_patient = {
    'female_age': 30,
    'male_age': 32,
    'bmi': 25.0,
    # ... rest of features
}
sample_patients.append(new_patient)
```

### Option 3: Batch Processing with Output

```python
import csv

results = []
for i, patient in enumerate(sample_patients, 1):
    features_scaled = prepare_features(patient, scaler)
    prediction, probabilities = predict(model, features_scaled, device)
    
    results.append({
        'patient_id': i,
        'prediction': 'Positive' if prediction == 1 else 'Negative',
        'confidence': max(probabilities) * 100,
        'prob_negative': probabilities[0] * 100,
        'prob_positive': probabilities[1] * 100
    })

# Save to CSV
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
```

---

## 📋 IUI Patient Data Format

Required columns for IUI predictions:

| Field | Type | Example Values |
|-------|------|-----------------|
| female_age | int | 22-44 |
| male_age | int | 24-54 |
| bmi | float | 18-40 |
| cycle_regular | string | Regular / Irregular |
| pcos | string | Yes / No |
| smoker_female | string | Yes / No |
| smoker_male | string | Yes / No |
| amh | float | 0.1-8.0 |
| fsh | float | 2-15 |
| sperm_count_million_ml | float | 10-100 |
| sperm_motility_percent | float | 20-70 |
| iui_cycle_type | string | Natural / Stimulated |
| follicle_count | int | 1-3 |
| trigger_shot | string | Yes / No |
| multiple_pregnancy_risk | string | Yes / No |
| ectopic_risk | string | Yes / No |
| trisomy21_risk | string | Yes / No |
| trisomy18_risk | string | Yes / No |
| endometriosis | string | None / Stage I / Stage II / Stage III / Stage IV |

**See:** `examples/iui_example_patients.csv` for 4 sample records

---

## 📋 IVF Patient Data Format

Required columns for IVF predictions:

| Field | Type | Example Values |
|-------|------|-----------------|
| female_age | int | 22-44 |
| male_age | int | 24-54 |
| bmi | float | 18-40 |
| ovarian_reserve | string | Very Low / Low / Normal / High |
| previous_ivf_failures | int | 0-3 |
| pcos | string | Yes / No |
| endometriosis | string | None / Stage I / Stage II / Stage III / Stage IV |
| amh | float | 0.1-8.0 |
| fsh | float | 2-15 |
| oocytes_retrieved | int | 1-20 |
| fertilization_method | string | IVF / ICSI |
| embryos_created | int | 1-10 |
| embryos_transferred | int | 1-2 |
| ohss_risk | string | Yes / No |
| multiple_pregnancy_risk | string | Yes / No |
| trisomy21_risk | string | Yes / No |
| trisomy18_risk | string | Yes / No |
| trisomy13_risk | string | Yes / No |
| implantation_failure_risk | string | Yes / No |

**See:** `examples/ivf_example_patients.csv` for 4 sample records

---

## 📈 Understanding Model Predictions

### Output Interpretation

```
Prediction: POSITIVE (Pregnancy likely)
Confidence: 78.45%
Probabilities - Negative: 21.5%, Positive: 78.5%
```

- **Prediction**: Binary outcome (POSITIVE or NEGATIVE)
- **Confidence**: How certain the model is (higher = more confident)
- **Probabilities**: Softmax probabilities for each class

### Confidence Levels

- **> 75%**: Very confident prediction
- **65-75%**: Moderately confident
- **55-65%**: Somewhat confident
- **< 55%**: Low confidence, treat with caution

---

## 🎯 Example Scenarios

### Scenario 1: Young, Healthy IUI Patient
```csv
female_age,male_age,bmi,cycle_regular,pcos,smoker_female,smoker_male,amh,fsh,sperm_count_million_ml,sperm_motility_percent,iui_cycle_type,follicle_count,trigger_shot,multiple_pregnancy_risk,ectopic_risk,trisomy21_risk,trisomy18_risk,endometriosis
28,30,24.5,Regular,No,No,No,2.5,5.0,50,45,Stimulated,2,Yes,No,No,No,No,None
```
**Expected**: HIGH success probability (70-80%)

### Scenario 2: Older IUI Patient with Complications
```csv
female_age,male_age,bmi,cycle_regular,pcos,smoker_female,smoker_male,amh,fsh,sperm_count_million_ml,sperm_motility_percent,iui_cycle_type,follicle_count,trigger_shot,multiple_pregnancy_risk,ectopic_risk,trisomy21_risk,trisomy18_risk,endometriosis
38,40,28.0,Irregular,Yes,No,Yes,1.0,10.0,30,30,Natural,1,No,No,Yes,Yes,No,Stage II
```
**Expected**: LOW success probability (20-30%)

### Scenario 3: Young IVF with Normal Reserve
```csv
female_age,male_age,bmi,ovarian_reserve,previous_ivf_failures,pcos,endometriosis,amh,fsh,oocytes_retrieved,fertilization_method,embryos_created,embryos_transferred,ohss_risk,multiple_pregnancy_risk,trisomy21_risk,trisomy18_risk,trisomy13_risk,implantation_failure_risk
28,30,24.0,Normal,0,No,None,3.0,6.0,15,IVF,10,2,No,Yes,No,No,No,No
```
**Expected**: MODERATE-HIGH success probability (60-75%)

### Scenario 4: Older IVF with Low Reserve and Failures
```csv
female_age,male_age,bmi,ovarian_reserve,previous_ivf_failures,pcos,endometriosis,amh,fsh,oocytes_retrieved,fertilization_method,embryos_created,embryos_transferred,ohss_risk,multiple_pregnancy_risk,trisomy21_risk,trisomy18_risk,trisomy13_risk,implantation_failure_risk
42,44,26.5,Very Low,2,Yes,Stage II,0.8,12.0,4,ICSI,2,2,No,No,Yes,No,No,Yes
```
**Expected**: LOW success probability (15-25%)

---

## ⚠️ Important Notes

### Model Accuracy
- **IUI PyTorch Model**: 73.70% test accuracy
- **IVF PyTorch Model**: 72.60% test accuracy

### Limitations
- Models are trained on **synthetic data** with realistic distributions
- Use for **research and decision support only**, not clinical diagnosis
- Always consult medical professionals for clinical decisions
- Models assume data quality and completeness

### Technical Requirements
- **PyTorch**: For neural network inference
- **Scikit-learn**: For feature scaling
- **Pandas**: For data loading
- **NumPy**: For numerical operations

### Device Support
- Scripts automatically detect GPU (CUDA) if available
- Falls back to CPU if GPU not available
- GPU inference is significantly faster for large batches

---

## 🔧 Troubleshooting

### "Model file not found at ../models/model_m_iui_pytorch.pth"

**Solution**: Ensure folder structure is:
```
parent_folder/
├── models/
│   ├── model_m_iui_pytorch.pth
│   ├── model_m_ivf_pytorch.pth
│   ├── iui_synthetic_dataset.csv
│   ├── ivf_synthetic_dataset.csv
│   └── ... (other model files)
└── inference_deployment/
    ├── iui_inference.py
    ├── ivf_inference.py
    ├── examples/
    └── README.md
```

### "Missing required columns in CSV"

**Solution**: Check that your CSV has ALL required columns listed above. Run the example first:
```bash
python iui_inference.py  # This uses built-in sample data, requires no CSV
```

### "Dimension mismatch" error

**Solution**: 
1. Verify all categorical variables use exact string values (Yes/No, not Y/N)
2. Check spelling matches examples (e.g., "Regular" not "regular")
3. Ensure no extra or missing columns in your data

### "CUDA out of memory"

**Solution**: The script automatically uses CPU if GPU memory is insufficient. This is slower but will work.

### "ValueError: could not convert string to float"

**Solution**: Ensure numeric columns contain only numbers:
- No text values in AMH, FSH, age fields
- No commas or special characters in numeric fields
- Check for blank cells in numeric columns

---

## 📚 File Descriptions

### iui_inference.py
- **Purpose**: Load IUI model and predict pregnancy success
- **Input**: Patient data with IUI-specific fields
- **Output**: Binary prediction + probability scores
- **Models Used**: `model_m_iui_pytorch.pth`
- **Lines**: ~305
- **Features**: 29 base + 10 engineered = 39 total

### ivf_inference.py
- **Purpose**: Load IVF model and predict pregnancy success
- **Input**: Patient data with IVF-specific fields
- **Output**: Binary prediction + probability scores
- **Models Used**: `model_m_ivf_pytorch.pth`
- **Lines**: ~305
- **Features**: 19 base + 13 engineered = 32 total

### examples/iui_example_patients.csv
- **Records**: 4 diverse IUI patient scenarios
- **Purpose**: Test and understand model predictions
- **Columns**: 19 IUI features

### examples/ivf_example_patients.csv
- **Records**: 4 diverse IVF patient scenarios
- **Purpose**: Test and understand model predictions
- **Columns**: 19 IVF features

---

## 🚀 Advanced Usage

### Custom Feature Engineering

If you want to modify feature engineering, edit the `prepare_features()` function in either script:

```python
def prepare_features(patient_data, scaler):
    # ... feature extraction code ...
    
    # Add custom features
    custom_feature = patient_data['amh'] * patient_data['female_age']
    engineered.append(custom_feature)
    
    # ... rest of function ...
```

### Model Fine-tuning

To retrain models on new data, see the training scripts in the `models/` folder:
- `model_m_iui_pytorch.py` - IUI training script
- `model_m_ivf_pytorch.py` - IVF training script

### Batch Processing Template

```python
import pandas as pd
from iui_inference import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('../models/model_m_iui_pytorch.pth', device)

df = pd.read_csv('examples/iui_example_patients.csv')
predictions = []

for idx, row in df.iterrows():
    features = prepare_features(row.to_dict(), scaler)
    pred, probs = predict(model, features, device)
    predictions.append({
        'patient': idx,
        'prediction': pred,
        'confidence': max(probs)
    })

results_df = pd.DataFrame(predictions)
results_df.to_csv('results.csv', index=False)
```

---

## 📞 Support

For issues with:
- **Model Accuracy**: Check data quality and ensure features match training data distributions
- **Installation**: Install PyTorch from https://pytorch.org/get-started/locally/
- **Data Format**: Use provided CSV examples as templates
- **Performance**: GPU inference 5-10x faster than CPU for large batches

---

**Version**: 1.0  
**Last Updated**: December 14, 2025  
**Compatibility**: Python 3.7+, PyTorch 1.9+, Scikit-learn 0.24+
