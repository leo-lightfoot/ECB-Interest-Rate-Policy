import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully.")

def check_data_files():
    """Check if necessary data files exist."""
    required_data_files = [
        os.path.join("Processed_Data", "ecb_meeting_data.csv")
    ]
    
    for file_path in required_data_files:
        if not os.path.exists(file_path):
            print(f"Warning: Required data file not found: {file_path}")
            if "ecb_meeting_data.csv" in file_path:
                print("You may need to run Data_Preprocessing.py first.")
        else:
            print(f"Found data file: {file_path}")

def check_model_files():
    """Check if necessary model files exist."""
    required_model_files = [
        os.path.join("models", "smote", "smote_rf.pkl")
    ]
    
    for file_path in required_model_files:
        if not os.path.exists(file_path):
            print(f"Warning: Required model file not found: {file_path}")
            if "smote_rf.pkl" in file_path:
                print("You may need to run model_training.py first to train the models.")
        else:
            print(f"Found model file: {file_path}")

def create_feature_names_file():
    """Create feature names JSON file if not exists."""
    feature_names_file = os.path.join("models", "feature_names.json")
    
    if not os.path.exists(feature_names_file):
        print("Feature names file not found. Attempting to create it...")
        
        # Check if the script exists
        script_path = os.path.join("models", "save_feature_names.py")
        if os.path.exists(script_path):
            # Change directory to models folder and run the script
            os.chdir("models")
            subprocess.check_call([sys.executable, "save_feature_names.py"])
            os.chdir("..")
            
            if os.path.exists(feature_names_file):
                print("Feature names file created successfully.")
            else:
                print("Error: Failed to create feature names file.")
        else:
            print(f"Error: Script not found: {script_path}")
    else:
        print(f"Found feature names file: {feature_names_file}")

def test_streamlit_app():
    """Test if Streamlit app runs without errors."""
    print("Testing Streamlit app...")
    try:
        # Run streamlit with --no-browser flag to prevent opening browser window
        subprocess.check_call([
            sys.executable, "-m", "streamlit", "run", "app.py", "--no-browser"
        ], timeout=5)
    except subprocess.TimeoutExpired:
        # This is expected as we're just checking if it starts without errors
        print("Streamlit app starts successfully.")
    except Exception as e:
        print(f"Error testing Streamlit app: {e}")

def main():
    print("Running setup for ECB Interest Rate Policy Prediction...")
    
    # Run checks
    check_python_version()
    install_requirements()
    check_data_files()
    check_model_files()
    create_feature_names_file()
    
    print("\nSetup complete. You can now run the Streamlit app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main() 