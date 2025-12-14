"""
IUI Model Inference Script
Uses trained PyTorch neural network to predict IUI treatment success
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
    def __init__(self, input_size=29, hidden_layers=None):
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
    model = FertilityNeuralNetwork(input_size=29)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def prepare_features(patient_data, scaler):
    """Engineer features from patient data"""
    # Get base features
    base_features = ['female_age', 'male_age', 'bmi', 'cycle_regular', 'pcos', 
                     'smoker_female', 'smoker_male', 'amh', 'fsh', 
                     'sperm_count_million_ml', 'sperm_motility_percent', 
                     'iui_cycle_type', 'follicle_count', 'trigger_shot',
                     'multiple_pregnancy_risk', 'ectopic_risk', 'trisomy21_risk', 
                     'trisomy18_risk', 'endometriosis']
    
    # Extract base values
    features = []
    for col in base_features:
        val = patient_data.get(col, 0)
        if isinstance(val, str):
            # Encode categorical variables
            if col in ['cycle_regular', 'pcos', 'smoker_female', 'smoker_male', 
                      'trigger_shot', 'multiple_pregnancy_risk', 'ectopic_risk',
                      'trisomy21_risk', 'trisomy18_risk']:
                val = 1 if val.lower() == 'yes' else 0
            elif col == 'iui_cycle_type':
                val = 1 if val.lower() == 'stimulated' else 0
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
    amh = features[7]
    fsh = features[8]
    sperm_count = features[9]
    sperm_motility = features[10]
    
    # Create engineered features
    age_interaction = female_age * amh
    hormone_ratio = fsh / (amh + 0.1)  # Avoid division by zero
    sperm_quality = sperm_count * sperm_motility / 100
    age_squared = female_age ** 2
    bmi_interaction = bmi * female_age
    fsh_amh_ratio = fsh / (amh + 0.1)
    sperm_motility_amh = sperm_motility * amh
    age_group = 1 if female_age < 35 else 0
    ovarian_reserve_score = (1 / (fsh + 0.1)) * amh
    
    engineered = np.array([
        age_interaction, hormone_ratio, sperm_quality, age_squared, 
        bmi_interaction, fsh_amh_ratio, sperm_motility_amh, age_group, 
        ovarian_reserve_score, sperm_count * amh
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
    model_path = os.path.join(models_dir, 'model_m_iui_pytorch.pth')
    dataset_path = os.path.join(models_dir, 'iui_synthetic_dataset.csv')
    
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
    base_features = ['female_age', 'male_age', 'bmi', 'cycle_regular', 'pcos', 
                     'smoker_female', 'smoker_male', 'amh', 'fsh', 
                     'sperm_count_million_ml', 'sperm_motility_percent', 
                     'iui_cycle_type', 'follicle_count', 'trigger_shot',
                     'multiple_pregnancy_risk', 'ectopic_risk', 'trisomy21_risk', 
                     'trisomy18_risk', 'endometriosis']
    
    # Encode categorical variables in training data
    df_train_encoded = df_train.copy()
    for col in ['cycle_regular', 'pcos', 'smoker_female', 'smoker_male', 
               'trigger_shot', 'multiple_pregnancy_risk', 'ectopic_risk',
               'trisomy21_risk', 'trisomy18_risk']:
        df_train_encoded[col] = (df_train[col].str.lower() == 'yes').astype(int)
    
    df_train_encoded['iui_cycle_type'] = (df_train['iui_cycle_type'].str.lower() == 'stimulated').astype(int)
    
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
        amh = row_dict['amh']
        fsh = row_dict['fsh']
        sperm_count = row_dict['sperm_count_million_ml']
        sperm_motility = row_dict['sperm_motility_percent']
        
        base_vals = [row_dict[col] for col in base_features]
        
        engineered = [
            female_age * amh,
            fsh / (amh + 0.1),
            sperm_count * sperm_motility / 100,
            female_age ** 2,
            bmi * female_age,
            fsh / (amh + 0.1),
            sperm_motility * amh,
            1 if female_age < 35 else 0,
            (1 / (fsh + 0.1)) * amh,
            sperm_count * amh
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
            'bmi': 24.5,
            'cycle_regular': 'Regular',
            'pcos': 'No',
            'smoker_female': 'No',
            'smoker_male': 'No',
            'amh': 2.5,
            'fsh': 5.0,
            'sperm_count_million_ml': 50,
            'sperm_motility_percent': 45,
            'iui_cycle_type': 'Stimulated',
            'follicle_count': 2,
            'trigger_shot': 'Yes',
            'multiple_pregnancy_risk': 'No',
            'ectopic_risk': 'No',
            'trisomy21_risk': 'No',
            'trisomy18_risk': 'No',
            'endometriosis': 'None'
        },
        {
            'female_age': 38,
            'male_age': 40,
            'bmi': 28.0,
            'cycle_regular': 'Irregular',
            'pcos': 'Yes',
            'smoker_female': 'No',
            'smoker_male': 'Yes',
            'amh': 1.0,
            'fsh': 10.0,
            'sperm_count_million_ml': 30,
            'sperm_motility_percent': 30,
            'iui_cycle_type': 'Natural',
            'follicle_count': 1,
            'trigger_shot': 'No',
            'multiple_pregnancy_risk': 'No',
            'ectopic_risk': 'No',
            'trisomy21_risk': 'Yes',
            'trisomy18_risk': 'No',
            'endometriosis': 'Stage II'
        }
    ]
    
    # Make predictions
    print("\n" + "="*60)
    print("IUI MODEL PREDICTIONS")
    print("="*60)
    
    for i, patient in enumerate(sample_patients, 1):
        print(f"\n--- Patient {i} ---")
        print(f"Age: {patient['female_age']}, Partner Age: {patient['male_age']}, BMI: {patient['bmi']}")
        print(f"PCOS: {patient['pcos']}, Cycle: {patient['cycle_regular']}, AMH: {patient['amh']}, FSH: {patient['fsh']}")
        print(f"Sperm Count: {patient['sperm_count_million_ml']} M/ml, Motility: {patient['sperm_motility_percent']}%")
        print(f"Follicles: {patient['follicle_count']}, Trigger: {patient['trigger_shot']}, Type: {patient['iui_cycle_type']}")
        
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
