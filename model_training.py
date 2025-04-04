import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
import shap

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create directories for outputs if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("models/standard", exist_ok=True)
os.makedirs("models/crisis_aware", exist_ok=True)
os.makedirs("models/smote", exist_ok=True)
os.makedirs("models/two_stage", exist_ok=True)
os.makedirs("models/weighted", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("crisis_analysis", exist_ok=True)
os.makedirs("shap_analysis", exist_ok=True)


# Define crisis periods
CRISIS_PERIODS = {
    "Global Financial Crisis": ["2008-01-01", "2009-12-31"],
    "European Debt Crisis": ["2010-04-01", "2012-07-31"],
    "COVID-19 Pandemic": ["2020-03-01", "2021-06-30"]
}

# Load the processed data
print("Loading data...")
# Change from original processed_data.csv to the new dataset
data = pd.read_csv('Processed_Data/ecb_meeting_data.csv')

# Convert date columns to datetime
data['Meeting_Date'] = pd.to_datetime(data['Meeting_Date'])

# Create target variable - Rate Hike (1), Hold (0), Rate Cut (-1)
data['ECB_rate_policy'] = np.sign(data['Next_Rate_Change'])

# Check if necessary columns exist
required_columns = ['Next_Rate_Change', 'Meeting_Date']
for col in required_columns:
    if col not in data.columns:
        print(f"Error: {col} column not found in the dataset")
        exit(1)
        
# Display basic statistics of the dataset
print(f"\nDataset shape: {data.shape}")
print(f"Number of meetings: {len(data)}")
print(f"Date range: {data['Meeting_Date'].min()} to {data['Meeting_Date'].max()}")
print(f"Target distribution:")
print(data['ECB_rate_policy'].value_counts())

# Process features for the new dataset format
print("\nProcessing features for the new dataset format...")

# Use interval information to create new features
data['Interval_Length'] = (pd.to_datetime(data['Interval_End']) - pd.to_datetime(data['Interval_Start'])).dt.days

# Create market volatility features
volatility_cols = [col for col in data.columns if '_volatility_' in col]
if volatility_cols:
    data['Market_Volatility_Avg'] = data[volatility_cols].mean(axis=1)
    print(f"Created Market_Volatility_Avg from {len(volatility_cols)} volatility columns")

# Create economic indicator features
inflation_cols = [col for col in data.columns if 'CPI' in col or 'Inflation' in col]

# Create key indicator averages
if inflation_cols:
    mean_inflation_cols = [col for col in inflation_cols if '_mean' in col]
    if mean_inflation_cols:
        data['Inflation_Indicators_Avg'] = data[mean_inflation_cols].mean(axis=1)
        print(f"Created Inflation_Indicators_Avg from {len(mean_inflation_cols)} inflation columns")

# Create yield curve features
if all(col in data.columns for col in ['slope_10y_2y_mean', 'slope_5y_1y_mean', 'curvature_mean']):
    data['Yield_Curve_Status'] = 0  # Neutral
    # Inverted yield curve (negative slope often signals recession)
    data.loc[data['slope_10y_2y_mean'] < -0.5, 'Yield_Curve_Status'] = -1
    # Steep yield curve (strongly positive slope)
    data.loc[data['slope_10y_2y_mean'] > 1.0, 'Yield_Curve_Status'] = 1
    print("Created Yield_Curve_Status feature")

# Calculate rate change lags (previous rate changes can be predictive)
data['Previous_Rate_Decision'] = data['Next_Rate_Change'].shift(1).fillna(0)
data['Two_Meetings_Ago_Decision'] = data['Next_Rate_Change'].shift(2).fillna(0)
print("Created rate decision lag features")

# Political Risk indicators
risk_cols = [col for col in data.columns if 'GPR' in col]
if risk_cols:
    mean_risk_cols = [col for col in risk_cols if '_mean' in col]
    if mean_risk_cols:
        data['Political_Risk_Avg'] = data[mean_risk_cols].mean(axis=1)
        print(f"Created Political_Risk_Avg from {len(mean_risk_cols)} geopolitical risk columns")

# Drop rows with NaN values due to lagging
initial_rows = len(data)
data = data.dropna().reset_index(drop=True)
print(f"Dropped {initial_rows - len(data)} rows with NaN values")
print(f"Data shape after creating features: {data.shape}")

# Identify crisis periods
print("\nIdentifying crisis periods...")
data['is_crisis'] = False
data['crisis_type'] = 'Normal Period'

for crisis_name, (start_date, end_date) in CRISIS_PERIODS.items():
    mask = (data['Meeting_Date'] >= start_date) & (data['Meeting_Date'] <= end_date)
    data.loc[mask, 'is_crisis'] = True
    data.loc[mask, 'crisis_type'] = crisis_name

print("\nDistribution of Rate Policy Actions by Period Type:")
period_policy_dist = pd.crosstab(data['crisis_type'], data['ECB_rate_policy'])
period_policy_dist.columns = ['Rate Cut (-1)', 'Hold (0)', 'Rate Hike (1)']
print(period_policy_dist)

plt.figure(figsize=(12, 6))
period_policy_dist.plot(kind='bar', stacked=True)
plt.title('ECB Rate Policy Decisions During Different Periods')
plt.xlabel('Period')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('crisis_analysis/policy_distribution_by_period.png')
plt.close()

crisis_freq = data['is_crisis'].mean()
normal_freq = 1 - crisis_freq
crisis_weight = normal_freq / (normal_freq + crisis_freq)
normal_weight = crisis_freq / (normal_freq + crisis_freq)

total = crisis_weight + normal_weight
crisis_weight /= total
normal_weight /= total

print(f"\nCrisis weight: {crisis_weight:.4f}")
print(f"Normal weight: {normal_weight:.4f}")

print("\nSetting up features and target...")
exclude_cols = ['Meeting_Date', 'Interval_Start', 'Interval_End', 'Date_x', 'Date_y', 'month',
                'Next_Rate_Change', 'ECB_rate_change', 'ECB_rate_acceleration', 
                'ECB_rate_policy', 'is_crisis', 'crisis_type']

X = data.drop(exclude_cols, axis=1, errors='ignore')

non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_cols:
    print(f"Warning: Removing non-numeric columns: {non_numeric_cols}")
    X = X.drop(non_numeric_cols, axis=1)

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

print(f"Number of features: {X.shape[1]}")
print(f"Feature types: {X.dtypes.value_counts()}")

y = data['ECB_rate_policy']

X_with_crisis = X.copy()
X_with_crisis['is_crisis'] = data['is_crisis'].astype(int)

def safe_classification_report(y_true, y_pred, output_dict=True):
    """Generate a classification report that handles missing classes."""
    unique_true = set(y_true)
    unique_pred = set(y_pred)
    all_classes = {-1.0, 0.0, 1.0}
    missing_classes = all_classes - (unique_true.union(unique_pred))
    if missing_classes:
        print(f"Warning: Missing classes in predictions or ground truth: {missing_classes}")
    return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)

def evaluate_model(model, X_test, y_test, model_type="standard"):
    y_pred = model.predict(X_test)
    
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None
    
    accuracy = accuracy_score(y_test, y_pred)
    
    y_test_binarized = {}
    y_pred_binarized = {}
    
    f1_scores = {}
    
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[-1.0, 0.0, 1.0])
    class_report = safe_classification_report(y_test, y_pred, output_dict=True)
    
    for cls in [-1.0, 0.0, 1.0]:
        cls_str = str(cls)
        y_test_binarized[cls] = (y_test == cls).astype(int)
        y_pred_binarized[cls] = (y_pred == cls).astype(int)
        f1_scores[cls] = f1_score(y_test_binarized[cls], y_pred_binarized[cls])
    
    plt.figure(figsize=(10, 8))
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
    
    labels = np.asarray(
        [
            [f"{conf_matrix[i, j]}\n({conf_matrix_norm[i, j]:.1%})" 
             for j in range(conf_matrix.shape[1])]
            for i in range(conf_matrix.shape[0])
        ]
    )
    
    sns.heatmap(conf_matrix, annot=labels, fmt="", cmap='Blues',
               xticklabels=['Rate Cut (-1)', 'Hold (0)', 'Rate Hike (1)'],
               yticklabels=['Rate Cut (-1)', 'Hold (0)', 'Rate Hike (1)'])
    plt.title(f'Confusion Matrix - {model_type} Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{model_type}.png')
    plt.close()
    
    if y_prob is not None:
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red']
        class_names = ['Rate Cut', 'Hold', 'Rate Hike']
        
        for i, cls in enumerate([-1.0, 0.0, 1.0]):
            cls_idx = i
            if cls_idx < y_prob.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_binarized[cls], y_prob[:, cls_idx])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                         label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_type} Model')
        plt.legend(loc="lower right")
        plt.savefig(f'plots/roc_curve_{model_type}.png')
        plt.close()
    
    if hasattr(model, 'feature_importances_'):
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top 20 Feature Importances - {model_type} Model')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f'plots/feature_importance_{model_type}.png')
            plt.close()
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'test_class_dist': pd.Series(y_test).value_counts().to_dict(),
        'predictions': y_pred,
        'test_y': y_test,
        'f1_scores': f1_scores
    }

def tune_hyperparameters(X_train, y_train, model_type="standard"):
    print(f"Tuning hyperparameters for {model_type} model...")
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }
    
    if model_type == "weighted":
        param_grid['class_weight'] = ['balanced', 'balanced_subsample']
        base_model = RandomForestClassifier(random_state=42)
    else:
        base_model = RandomForestClassifier(random_state=42)
    
    if model_type == "smote":
        try:
            smote_tomek = SMOTETomek(random_state=42, sampling_strategy='auto')
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
            X_train = X_train_resampled
            y_train = y_train_resampled
            print("Applied SMOTETomek for improved class balance")
        except Exception as e:
            print(f"SMOTETomek failed: {e}")
            print("Falling back to regular SMOTE")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_type} model: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_params_

# Split data into training and testing sets (80/20 split)
print("\nSplitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create crisis-aware training and testing features
X_train_with_crisis = X_with_crisis.loc[X_train.index]
X_test_with_crisis = X_with_crisis.loc[X_test.index]

# Print class distribution
print("\nClass distribution in training set:")
print(Counter(y_train))
print("\nClass distribution in test set:")
print(Counter(y_test))

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Train and evaluate models
print("\nTraining models...")
all_results = []

# 1. Standard Random Forest model
print("\nTraining standard Random Forest model...")
use_hyperparameter_tuning = True

if use_hyperparameter_tuning:
    try:
        best_params = tune_hyperparameters(X_train_scaled, y_train, "standard")
        standard_model = RandomForestClassifier(**best_params, random_state=42)
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Falling back to default parameters")
        standard_model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    standard_model = RandomForestClassifier(n_estimators=100, random_state=42)

standard_model.fit(X_train_scaled, y_train)
standard_result = evaluate_model(standard_model, X_test_scaled, y_test, "Standard")
all_results.append(standard_result)
joblib.dump(standard_model, "models/standard/standard_rf.pkl")

# 2. Crisis-aware Random Forest model
print("\nTraining crisis-aware Random Forest model...")
if use_hyperparameter_tuning:
    try:
        best_params = tune_hyperparameters(X_train_with_crisis, y_train, "crisis_aware")
        crisis_model = RandomForestClassifier(**best_params, random_state=42)
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Falling back to default parameters")
        crisis_model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    crisis_model = RandomForestClassifier(n_estimators=100, random_state=42)

crisis_model.fit(X_train_with_crisis, y_train)
crisis_result = evaluate_model(crisis_model, X_test_with_crisis, y_test, "Crisis-Aware")
all_results.append(crisis_result)
joblib.dump(crisis_model, "models/crisis_aware/crisis_aware_rf.pkl")

# 3. Weighted Random Forest model
print("\nTraining weighted Random Forest model...")
if use_hyperparameter_tuning:
    try:
        best_params = tune_hyperparameters(X_train_scaled, y_train, "weighted")
        weighted_model = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Falling back to default parameters")
        weighted_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
else:
    weighted_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

weighted_model.fit(X_train_scaled, y_train)
weighted_result = evaluate_model(weighted_model, X_test_scaled, y_test, "Weighted")
all_results.append(weighted_result)
joblib.dump(weighted_model, "models/weighted/weighted_rf.pkl")

# 4. SMOTE model
print("\nTraining SMOTE model...")
min_samples = min(pd.Series(y_train).value_counts())
if min_samples >= 3:
    k_neighbors = min(4, min_samples - 1)
    print(f"Using k_neighbors={k_neighbors} for SMOTE (min_samples={min_samples})")
    
    if use_hyperparameter_tuning:
        try:
            best_params = tune_hyperparameters(X_train_scaled, y_train, "smote")
            smote_model = RandomForestClassifier(**best_params, random_state=42)
            smote_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            print("Falling back to default parameters")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            smote_model = RandomForestClassifier(n_estimators=100, random_state=42)
            smote_model.fit(X_train_smote, y_train_smote)
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        smote_model = RandomForestClassifier(n_estimators=100, random_state=42)
        smote_model.fit(X_train_smote, y_train_smote)
    
    smote_result = evaluate_model(smote_model, X_test_scaled, y_test, "SMOTE")
    all_results.append(smote_result)
    joblib.dump(smote_model, "models/smote/smote_rf.pkl")
    
    print("\nApplying custom probability thresholds for SMOTE model...")
    try:
        def apply_custom_threshold(model, X_test, thresholds={0: 0.3, 1: 0.4, 2: 0.3}):
            proba = model.predict_proba(X_test)
            custom_preds = np.zeros(len(X_test))
            
            for i in range(len(X_test)):
                adjusted_proba = [proba[i, j] / thresholds[j] for j in range(3)]
                class_index = np.argmax(adjusted_proba)
                custom_preds[i] = model.classes_[class_index]
            
            return custom_preds
            
        custom_preds = apply_custom_threshold(smote_model, X_test_scaled)
        
        custom_accuracy = accuracy_score(y_test, custom_preds)
        custom_f1_scores = {}
        for cls in [-1.0, 0.0, 1.0]:
            y_test_binary = (y_test == cls).astype(int)
            y_pred_binary = (custom_preds == cls).astype(int)
            custom_f1_scores[cls] = f1_score(y_test_binary, y_pred_binary)
            
        print(f"Custom threshold accuracy: {custom_accuracy:.4f}")
        print(f"Custom threshold F1 scores: Rate Cut={custom_f1_scores[-1.0]:.4f}, Hold={custom_f1_scores[0.0]:.4f}, Rate Hike={custom_f1_scores[1.0]:.4f}")
        
        if custom_accuracy > smote_result['accuracy']:
            print("Custom threshold improved accuracy - saving as separate model")
            custom_result = {
                'model_type': 'SMOTE_Custom_Threshold',
                'accuracy': custom_accuracy,
                'confusion_matrix': confusion_matrix(y_test, custom_preds, labels=[-1.0, 0.0, 1.0]),
                'classification_report': safe_classification_report(y_test, custom_preds),
                'test_class_dist': pd.Series(y_test).value_counts().to_dict(),
                'predictions': custom_preds,
                'test_y': y_test,
                'f1_scores': custom_f1_scores
            }
            all_results.append(custom_result)
    except Exception as e:
        print(f"Custom threshold application failed: {e}")
        
else:
    print(f"Skipping SMOTE as minimum class count ({min_samples}) is too small")

# 5. Two-Stage model
print("\nTraining Two-Stage model...")
# First stage: binary classification (change vs no change)
y_train_binary = (y_train != 0).astype(int)
binary_model = RandomForestClassifier(n_estimators=100, random_state=42)
binary_model.fit(X_train_scaled, y_train_binary)

# Second stage: direction classification (up vs down)
change_mask = y_train != 0
y_train_direction = (y_train[change_mask] > 0).astype(int)

if len(y_train_direction) > 0 and len(np.unique(y_train_direction)) > 1:
    direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
    direction_model.fit(X_train_scaled.iloc[change_mask.values], y_train_direction)
    
    binary_preds = binary_model.predict(X_test_scaled)
    two_stage_preds = np.zeros_like(y_test)
    change_idx = np.where(binary_preds == 1)[0]
    if len(change_idx) > 0:
        direction_preds = direction_model.predict(X_test_scaled.iloc[change_idx])
        two_stage_preds[change_idx] = np.where(direction_preds == 0, -1, 1)
    
    # Create confusion matrix plot for Two-Stage model
    plt.figure(figsize=(8, 6))
    two_stage_conf_matrix = confusion_matrix(y_test, two_stage_preds, labels=[-1.0, 0.0, 1.0])
    two_stage_conf_matrix_norm = two_stage_conf_matrix.astype('float') / two_stage_conf_matrix.sum(axis=1)[:, np.newaxis]
    two_stage_conf_matrix_norm = np.nan_to_num(two_stage_conf_matrix_norm)
    
    labels = np.asarray(
        [
            [f"{two_stage_conf_matrix[i, j]}\n({two_stage_conf_matrix_norm[i, j]:.1%})" 
             for j in range(two_stage_conf_matrix.shape[1])]
            for i in range(two_stage_conf_matrix.shape[0])
        ]
    )
    
    sns.heatmap(two_stage_conf_matrix, annot=labels, fmt="", cmap='Blues',
               xticklabels=['Rate Cut (-1)', 'Hold (0)', 'Rate Hike (1)'],
               yticklabels=['Rate Cut (-1)', 'Hold (0)', 'Rate Hike (1)'])
    plt.title('Confusion Matrix - Two_Stage Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_Two_Stage.png')
    plt.close()
    
    two_stage_class_report = safe_classification_report(y_test, two_stage_preds)
    
    # Calculate F1 scores for each class
    two_stage_f1_scores = {}
    for cls in [-1.0, 0.0, 1.0]:
        y_test_binary = (y_test == cls).astype(int)
        y_pred_binary = (two_stage_preds == cls).astype(int)
        two_stage_f1_scores[cls] = f1_score(y_test_binary, y_pred_binary)
    
    two_stage_result = {
        'model_type': 'Two_Stage',
        'accuracy': accuracy_score(y_test, two_stage_preds),
        'confusion_matrix': two_stage_conf_matrix,
        'classification_report': two_stage_class_report,
        'test_class_dist': pd.Series(y_test).value_counts().to_dict(),
        'predictions': two_stage_preds,
        'test_y': y_test,
        'f1_scores': two_stage_f1_scores
    }
    
    all_results.append(two_stage_result)
    joblib.dump(binary_model, "models/two_stage/binary_rf.pkl")
    joblib.dump(direction_model, "models/two_stage/direction_rf.pkl")

# Save model comparison results
print("\nSaving model comparison results...")

def extract_metric(report, class_label, metric, default_value=np.nan):
    """Safely extract a metric from classification report for a given class."""
    if str(class_label) in report:
        return report[str(class_label)].get(metric, default_value)
    return default_value

# Create a dataframe with all results
results_df = pd.DataFrame([
    {
        'model_type': r['model_type'],
        'accuracy': r['accuracy'],
        'precision_rate_cut': extract_metric(r['classification_report'], '-1.0', 'precision'),
        'recall_rate_cut': extract_metric(r['classification_report'], '-1.0', 'recall'),
        'f1_rate_cut': extract_metric(r['classification_report'], '-1.0', 'f1-score'),
        'precision_hold': extract_metric(r['classification_report'], '0.0', 'precision'),
        'recall_hold': extract_metric(r['classification_report'], '0.0', 'recall'),
        'f1_hold': extract_metric(r['classification_report'], '0.0', 'f1-score'),
        'precision_rate_hike': extract_metric(r['classification_report'], '1.0', 'precision'),
        'recall_rate_hike': extract_metric(r['classification_report'], '1.0', 'recall'),
        'f1_rate_hike': extract_metric(r['classification_report'], '1.0', 'f1-score')
    }
    for r in all_results
])

# Save results to CSV
results_df.to_csv('results/model_performance_summary.csv', index=False)

# Create comparison table for easier viewing
comparison_table = results_df[['model_type', 'accuracy', 'f1_rate_cut', 'f1_hold', 'f1_rate_hike']]
comparison_table = comparison_table.sort_values('accuracy', ascending=False)
comparison_table.to_csv('results/final_model_comparison.csv', index=False)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': standard_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.to_csv('results/feature_importance.csv', index=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance_top20.png')
plt.close()

# Create performance comparison plot
plt.figure(figsize=(14, 7))
plt.bar(results_df['model_type'], results_df['accuracy'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model Type')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('plots/model_comparison_accuracy.png')
plt.close()

# SHAP analysis
print("\nPerforming SHAP analysis...")

# Find the best performing model for SHAP analysis
best_model_name = comparison_table.iloc[0]['model_type']
print(f"Using {best_model_name} model for SHAP analysis as it had the best performance")

if best_model_name == 'Standard':
    shap_model = standard_model
    shap_data = X_test_scaled
elif best_model_name == 'Crisis-Aware':
    shap_model = crisis_model
    shap_data = X_test_with_crisis
elif best_model_name == 'SMOTE':
    shap_model = smote_model
    shap_data = X_test_scaled
elif best_model_name == 'SMOTE_Custom_Threshold' and 'smote_model' in locals():
    shap_model = smote_model  # Still use the SMOTE model for SHAP, as custom threshold just changes prediction rules
    shap_data = X_test_scaled
    print("Using SMOTE model for SHAP analysis (custom threshold just modifies prediction rules)")
elif best_model_name == 'Weighted':
    shap_model = weighted_model
    shap_data = X_test_scaled
elif best_model_name == 'Two_Stage':
    # For Two-Stage, use the binary model for SHAP analysis
    shap_model = binary_model
    shap_data = X_test_scaled
    print("Note: For Two-Stage model, using binary classifier (change vs no change) for SHAP analysis")
else:
    # Default to using standard model
    shap_model = standard_model
    shap_data = X_test_scaled
    print("Using standard model for SHAP analysis")

# Limit the number of samples for SHAP analysis to avoid memory issues
max_shap_samples = min(100, len(shap_data))
shap_sample = shap_data.sample(max_shap_samples, random_state=42) if max_shap_samples < len(shap_data) else shap_data

# Create SHAP explainer
explainer = shap.TreeExplainer(shap_model)

# Calculate SHAP values
shap_values = explainer.shap_values(shap_sample)

# Global SHAP summary plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, shap_sample, show=False)
plt.title('SHAP Feature Importance - Global Impact on Model Output')
plt.tight_layout()
plt.savefig('shap_analysis/shap_summary_plot.png')
plt.close()

# Bar plot version of the summary
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance - Mean Absolute Impact')
plt.tight_layout()
plt.savefig('shap_analysis/shap_summary_bar_plot.png')
plt.close()

# Class-specific feature importance
class_names = ['Rate Cut (-1)', 'Hold (0)', 'Rate Hike (1)']
for i, class_name in enumerate(class_names):
    if isinstance(shap_values, list) and len(shap_values) > i:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values[i], shap_sample, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance for {class_name}')
        plt.tight_layout()
        plt.savefig(f'shap_analysis/feature_importance_{class_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
        plt.close()

# Generate SHAP dependence plots for top features
feature_importance = pd.DataFrame(shap_model.feature_importances_, index=shap_data.columns)
feature_importance.columns = ['importance']
feature_importance = feature_importance.sort_values('importance', ascending=False)
top_features = feature_importance.head(5).index.tolist()

for feature in top_features:
    if feature in shap_sample.columns:
        plt.figure(figsize=(10, 8))
        feature_idx = list(shap_sample.columns).index(feature)
        if isinstance(shap_values, list):
            # For classification, use the class with the most interesting pattern
            for i, class_name in enumerate(class_names):
                try:
                    shap.dependence_plot(feature_idx, shap_values[i], shap_sample, 
                                        show=False, title=f"{feature} impact on {class_name}")
                    plt.savefig(f'shap_analysis/dependence_plot_{feature}_{class_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating dependence plot for {feature} and class {class_name}: {e}")
        else:
            # For regression or single output
            try:
                shap.dependence_plot(feature_idx, shap_values, shap_sample, show=False)
                plt.title(f'How {feature} impacts predictions')
                plt.savefig(f'shap_analysis/dependence_plot_{feature}.png')
                plt.close()
            except Exception as e:
                print(f"Error creating dependence plot for {feature}: {e}")

print(f"SHAP analysis completed. Results saved to 'shap_analysis' directory.")
print("Top 5 most important features according to SHAP analysis:")
for i, feature in enumerate(top_features[:5]):
    print(f"{i+1}. {feature}")

# Save the SHAP explainer and values for later use
joblib.dump(explainer, "shap_analysis/shap_explainer.pkl")
np.save("shap_analysis/shap_values.npy", shap_values)

# Check feature importance of crisis indicator if it exists
if 'is_crisis' in X_with_crisis.columns:
    crisis_importance = crisis_model.feature_importances_[-1]
    print(f"\nImportance of 'is_crisis' feature in Crisis-Aware model: {crisis_importance:.6f}")

print("\nTraining complete! Results saved to files.")
print("Results are saved in the 'results' directory.")
print("Plots are saved in the 'plots' directory.")
print("Models are saved in the 'models' directory.")
print("SHAP analysis results are saved in the 'shap_analysis' directory.")

# Display results for all models
print("\nModel Results Summary:")
print("============================")
print(comparison_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nDetailed Model Performance:")
for i, row in results_df.iterrows():
    print(f"\nModel: {row['model_type']}")
    print(f"  Accuracy: {row['accuracy']:.4f}")
    print(f"  Rate Cut (-1) | Precision: {row['precision_rate_cut']:.4f} | Recall: {row['recall_rate_cut']:.4f} | F1: {row['f1_rate_cut']:.4f}")
    print(f"  Hold (0)      | Precision: {row['precision_hold']:.4f} | Recall: {row['recall_hold']:.4f} | F1: {row['f1_hold']:.4f}")
    print(f"  Rate Hike (1) | Precision: {row['precision_rate_hike']:.4f} | Recall: {row['recall_rate_hike']:.4f} | F1: {row['f1_rate_hike']:.4f}")

# Find best models for different purposes
print("\nBest Models for Different Purposes:")
print("=========================================")

if not results_df.empty:
    # Best for Rate Cut
    best_rate_cut_model = results_df.loc[results_df['f1_rate_cut'].idxmax()]
    print(f"Best for Rate Cut: {best_rate_cut_model['model_type']} (F1: {best_rate_cut_model['f1_rate_cut']:.4f})")

    # Best for Rate Hike
    best_rate_hike_model = results_df.loc[results_df['f1_rate_hike'].idxmax()]
    print(f"Best for Rate Hike: {best_rate_hike_model['model_type']} (F1: {best_rate_hike_model['f1_rate_hike']:.4f})")

    # Best overall accuracy
    best_acc_model = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"Best overall accuracy: {best_acc_model['model_type']} (Accuracy: {best_acc_model['accuracy']:.4f})")
    
    print("\nRecommended approach for production use:")
    print(f"1. Use {best_acc_model['model_type']} model for general predictions")
    print(f"2. Use {best_rate_cut_model['model_type']} model when sensitivity to Rate Cut events is critical")
    print(f"3. Use {best_rate_hike_model['model_type']} model when sensitivity to Rate Hike events is critical")
else:
    print("No model results found.") 