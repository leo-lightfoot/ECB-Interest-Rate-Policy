import joblib
import json
import os

# Path to the SMOTE model (using absolute path)
model_path = os.path.join('models', 'smote', 'smote_rf.pkl')

# Check if the model file exists
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Extract feature names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
        print(f"Extracted {len(feature_names)} feature names")
        
        # Save feature names to JSON in the models directory
        output_path = os.path.join('models', 'feature_names.json')
        with open(output_path, 'w') as f:
            json.dump(feature_names, f)
        
        print(f"Feature names saved to {output_path}")
    else:
        print("Error: Model doesn't have feature_names_in_ attribute")
else:
    print(f"Error: Model file not found at {model_path}") 