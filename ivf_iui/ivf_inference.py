"""
IVF Model Inference Script
Uses trained PyTorch neural network to predict IVF treatment success
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to path to access models folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Neural Network Architecture
class FertilityNeuralNetwork(nn.Module):
    def __init__(self, input_size=32, hidden_layers=None):
        super(FertilityNeuralNetwork, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32, 16]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 2))  # Binary classification
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_model(model_path, device):
    """Load trained model from .pth file"""
    model = FertilityNeuralNetwork(input_size=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def prepare_features(patient_data, scaler):
    """Engineer features from patient data"""
    # Get base features
    base_features = ['female_age', 'male_age', 'bmi', 'ovarian_reserve', 
                     'previous_ivf_failures', 'pcos', 'endometriosis', 'amh', 'fsh',
                     'oocytes_retrieved', 'fertilization_method', 'embryos_created',
                     'embryos_transferred', 'ohss_risk', 'multiple_pregnancy_risk',
                     'trisomy21_risk', 'trisomy18_risk', 'trisomy13_risk',
                     'implantation_failure_risk']
    
    # Extract base values
    features = []
    for col in base_features:
        val = patient_data.get(col, 0)
        if isinstance(val, str):
            # Encode categorical variables
            if col in ['pcos', 'ohss_risk', 'multiple_pregnancy_risk', 
                      'trisomy21_risk', 'trisomy18_risk', 'trisomy13_risk',
                      'implantation_failure_risk']:
                val = 1 if val.lower() == 'yes' else 0
            elif col == 'ovarian_reserve':
                reserve_map = {'very low': 0, 'low': 1, 'normal': 2, 'high': 3}
                val = reserve_map.get(val.lower(), 0)
            elif col == 'fertilization_method':
                val = 1 if val.lower() == 'icsi' else 0
            elif col == 'endometriosis':
                endometriosis_map = {'none': 0, 'stage i': 1, 'stage ii': 2,
                                    'stage iii': 3, 'stage iv': 4}
                val = endometriosis_map.get(val.lower(), 0)
        features.append(float(val))
    
    features_array = np.array(features).reshape(1, -1)
    
    # Engineer additional features
    female_age = features[0]
    male_age = features[1]
    bmi = features[2]
    previous_failures = features[4]
    amh = features[7]
    fsh = features[8]
    oocytes = features[9]
    embryos_created = features[11]
    
    # Create engineered features
    age_interaction = female_age * amh
    hormone_ratio = fsh / (amh + 0.1)
    oocyte_fertilization_rate = embryos_created / (oocytes + 0.1)
    age_squared = female_age ** 2
    bmi_interaction = bmi * female_age
    failure_penalty = 1 / (1 + previous_failures)  # Penalize failures
    fsh_amh_ratio = fsh / (amh + 0.1)
    age_ovarian_reserve = female_age * amh
    embryo_quality_score = embryos_created / (oocytes + 0.1)
    ovarian_reserve_score = (1 / (fsh + 0.1)) * amh
    age_group = 1 if female_age < 35 else 0
    
    age_fsh_interaction = female_age * fsh
    embryo_transfer_ratio = embryos_created / (fsh + 0.1)
    
    engineered = np.array([
        age_interaction, hormone_ratio, oocyte_fertilization_rate, age_squared,
        bmi_interaction, failure_penalty, fsh_amh_ratio, age_ovarian_reserve,
        embryo_quality_score, ovarian_reserve_score, age_group, age_fsh_interaction,
        embryo_transfer_ratio
    ]).reshape(1, -1)
    
    # Combine features
    all_features = np.hstack([features_array, engineered])
    
    # Scale features
    all_features_scaled = scaler.transform(all_features)
    
    return all_features_scaled


def predict(model, features_scaled, device):
    """Make prediction with model"""
    X_tensor = torch.FloatTensor(features_scaled).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
    
    return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]


def main():
    """Main inference function"""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    model_path = os.path.join(models_dir, 'model_m_ivf_pytorch.pth')
    dataset_path = os.path.join(models_dir, 'ivf_synthetic_dataset.csv')
    
    # Load model
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    model = load_model(model_path, device)
    
    # Load training data for scaler
    print(f"Loading training data from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file not found at {dataset_path}")
        return
    
    df_train = pd.read_csv(dataset_path)
    
    # Prepare scaler
    base_features = ['female_age', 'male_age', 'bmi', 'ovarian_reserve', 
                     'previous_ivf_failures', 'pcos', 'endometriosis', 'amh', 'fsh',
                     'oocytes_retrieved', 'fertilization_method', 'embryos_created',
                     'embryos_transferred', 'ohss_risk', 'multiple_pregnancy_risk',
                     'trisomy21_risk', 'trisomy18_risk', 'trisomy13_risk',
                     'implantation_failure_risk']
    
    # Encode categorical variables in training data
    df_train_encoded = df_train.copy()
    for col in ['pcos', 'ohss_risk', 'multiple_pregnancy_risk', 
               'trisomy21_risk', 'trisomy18_risk', 'trisomy13_risk',
               'implantation_failure_risk']:
        df_train_encoded[col] = (df_train[col].str.lower() == 'yes').astype(int)
    
    reserve_map = {'very low': 0, 'low': 1, 'normal': 2, 'high': 3}
    df_train_encoded['ovarian_reserve'] = df_train['ovarian_reserve'].map(
        lambda x: reserve_map.get(str(x).lower(), 0)
    )
    
    df_train_encoded['fertilization_method'] = (df_train['fertilization_method'].str.lower() == 'icsi').astype(int)
    
    endometriosis_map = {'none': 0, 'stage i': 1, 'stage ii': 2, 'stage iii': 3, 'stage iv': 4}
    df_train_encoded['endometriosis'] = df_train['endometriosis'].map(
        lambda x: endometriosis_map.get(str(x).lower(), 0) if pd.notna(x) else 0
    )
    
    # Build features with engineering for all training data
    features_list = []
    for idx, row in df_train_encoded.iterrows():
        row_dict = row.to_dict()
        female_age = row_dict['female_age']
        male_age = row_dict['male_age']
        bmi = row_dict['bmi']
        previous_failures = row_dict['previous_ivf_failures']
        amh = row_dict['amh']
        fsh = row_dict['fsh']
        oocytes = row_dict['oocytes_retrieved']
        embryos_created = row_dict['embryos_created']
        
        base_vals = [row_dict[col] for col in base_features]
        
        engineered = [
            female_age * amh,
            fsh / (amh + 0.1),
            embryos_created / (oocytes + 0.1),
            female_age ** 2,
            bmi * female_age,
            1 / (1 + previous_failures),
            fsh / (amh + 0.1),
            female_age * amh,
            embryos_created / (oocytes + 0.1),
            (1 / (fsh + 0.1)) * amh,
            1 if female_age < 35 else 0,
            female_age * fsh,
            embryos_created / (fsh + 0.1)
        ]
        
        features_list.append(base_vals + engineered)
    
    X_train = np.array(features_list)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Sample patients for demonstration
    sample_patients = [
        {
            'female_age': 28,
            'male_age': 30,
            'bmi': 24.0,
            'ovarian_reserve': 'Normal',
            'previous_ivf_failures': 0,
            'pcos': 'No',
            'endometriosis': 'None',
            'amh': 3.0,
            'fsh': 6.0,
            'oocytes_retrieved': 15,
            'fertilization_method': 'IVF',
            'embryos_created': 10,
            'embryos_transferred': 2,
            'ohss_risk': 'No',
            'multiple_pregnancy_risk': 'Yes',
            'trisomy21_risk': 'No',
            'trisomy18_risk': 'No',
            'trisomy13_risk': 'No',
            'implantation_failure_risk': 'No'
        },
        {
            'female_age': 42,
            'male_age': 44,
            'bmi': 26.5,
            'ovarian_reserve': 'Low',
            'previous_ivf_failures': 2,
            'pcos': 'No',
            'endometriosis': 'Stage II',
            'amh': 0.8,
            'fsh': 12.0,
            'oocytes_retrieved': 4,
            'fertilization_method': 'ICSI',
            'embryos_created': 2,
            'embryos_transferred': 2,
            'ohss_risk': 'No',
            'multiple_pregnancy_risk': 'No',
            'trisomy21_risk': 'Yes',
            'trisomy18_risk': 'No',
            'trisomy13_risk': 'No',
            'implantation_failure_risk': 'Yes'
        }
    ]
    
    # Make predictions
    print("\n" + "="*60)
    print("IVF MODEL PREDICTIONS")
    print("="*60)
    
    for i, patient in enumerate(sample_patients, 1):
        print(f"\n--- Patient {i} ---")
        print(f"Age: {patient['female_age']}, Partner Age: {patient['male_age']}, BMI: {patient['bmi']}")
        print(f"Ovarian Reserve: {patient['ovarian_reserve']}, Previous Failures: {patient['previous_ivf_failures']}")
        print(f"AMH: {patient['amh']}, FSH: {patient['fsh']}, Oocytes: {patient['oocytes_retrieved']}")
        print(f"Embryos Created: {patient['embryos_created']}, Transferred: {patient['embryos_transferred']}, Method: {patient['fertilization_method']}")
        
        features_scaled = prepare_features(patient, scaler)
        prediction, probabilities = predict(model, features_scaled, device)
        
        pred_label = "POSITIVE (Pregnancy likely)" if prediction == 1 else "NEGATIVE (Pregnancy unlikely)"
        confidence = max(probabilities) * 100
        
        print(f"\nPrediction: {pred_label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities - Negative: {probabilities[0]*100:.1f}%, Positive: {probabilities[1]*100:.1f}%")
    
    print("\n" + "="*60)
    print("To use with your own data, modify the sample_patients list")
    print("or load from CSV file using: pd.read_csv('your_file.csv')")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
