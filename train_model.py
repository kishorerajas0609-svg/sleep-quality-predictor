
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import preprocess_dataframe, FEATURE_COLUMNS, REVERSE_TARGET_MAP

def generate_synthetic_dataset(num_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    records = []
    caffeine_options = ['None', 'Low', 'Moderate', 'High']
    mood_options = ['Happy', 'Neutral', 'Sad', 'Anxious']
    interruptions_options = ['No', 'Yes']

    for _ in range(num_samples):
        stress = int(np.random.choice(range(0, 11), p=[0.05, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02]))
        screen = int(np.clip(np.random.normal(loc=45 + stress * 8, scale=25), 0, 180))
        exercise = int(np.clip(np.random.normal(loc=45 - stress * 3, scale=25), 0, 120))
        
        bed_h = float(np.clip(np.random.normal(loc=23.0 + (stress / 5.0), scale=1.2), 21.0, 27.0))
        sleep_dur = float(np.clip(np.random.normal(loc=7.8 - (stress * 0.25) - (screen / 120.0), scale=1.1), 4.0, 10.0))
        wake_h = (bed_h + sleep_dur) % 24.0
        actual_bed_h = bed_h % 24.0

        bedtime_str = f"{int(actual_bed_h):02d}:{int((actual_bed_h % 1) * 60):02d}"
        wakeup_str = f"{int(wake_h):02d}:{int((wake_h % 1) * 60):02d}"

        if stress >= 7:
            caffeine = np.random.choice(caffeine_options, p=[0.1, 0.2, 0.4, 0.3])
            mood = np.random.choice(mood_options, p=[0.1, 0.2, 0.35, 0.35])
            interruptions = np.random.choice(interruptions_options, p=[0.3, 0.7])
        elif stress <= 3:
            caffeine = np.random.choice(caffeine_options, p=[0.4, 0.4, 0.15, 0.05])
            mood = np.random.choice(mood_options, p=[0.6, 0.3, 0.05, 0.05])
            interruptions = np.random.choice(interruptions_options, p=[0.85, 0.15])
        else:
            caffeine = np.random.choice(caffeine_options, p=[0.25, 0.4, 0.25, 0.1])
            mood = np.random.choice(mood_options, p=[0.3, 0.4, 0.15, 0.15])
            interruptions = np.random.choice(interruptions_options, p=[0.6, 0.4])

        score = (
            (sleep_dur * 8.0) 
            - (stress * 4.5) 
            - (screen * 0.15) 
            + (exercise * 0.2) 
            - (15 if interruptions == 'Yes' else 0)
            - (10 if caffeine in ['Moderate', 'High'] else 0)
            + (10 if mood == 'Happy' else (-10 if mood in ['Sad', 'Anxious'] else 0))
            + np.random.normal(0, 4.0)
        )

        if score >= 42:
            quality = 'Good'
        elif score >= 24:
            quality = 'Average'
        else:
            quality = 'Poor'

        records.append({
            'sleep_duration': round(sleep_dur, 2),
            'bedtime': bedtime_str,
            'wakeup_time': wakeup_str,
            'caffeine_intake': caffeine,
            'exercise_duration': exercise,
            'screen_time': screen,
            'stress_level': stress,
            'mood': mood,
            'sleep_interruptions': interruptions,
            'sleep_quality': quality
        })
    return pd.DataFrame(records)

def train_and_evaluate_models():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    dataset_path = os.path.join('data', 'sleep_dataset.csv')
    print("--> Preparing Sleep Quality Dataset...")
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"Loaded existing dataset with {len(df)} samples.")
    else:
        df = generate_synthetic_dataset(num_samples=600)
        df.to_csv(dataset_path, index=False)
        print(f"Generated new synthetic dataset with {len(df)} samples and saved to '{dataset_path}'.")

    X_scaled, y, scaler = preprocess_dataframe(df, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split complete: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True, random_state=42)
    }

    all_metrics = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_obj = None

    print("\n" + "="*70)
    print(f"{'MODEL EVALUATION COMPARISON':^70}")
    print("="*70)
    for name, model in models.items():
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        all_metrics[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }

        print(f"Algorithm: {name}")
        print(f"  - Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  - Precision: {prec:.4f}")
        print(f"  - Recall:    {rec:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print("-" * 70)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model_obj = model

    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print(f"\n[BEST] BEST PERFORMING MODEL: {best_model_name} (Macro F1-Score: {best_f1_score:.4f})")

    model_bundle = {
        'model': best_model_obj,
        'scaler': scaler,
        'feature_names': FEATURE_COLUMNS,
        'target_mapping': REVERSE_TARGET_MAP,
        'model_name': best_model_name,
        'all_metrics': all_metrics
    }

    bundle_path = os.path.join('models', 'best_model.pkl')
    joblib.dump(model_bundle, bundle_path)
    print(f"Successfully saved best model bundle to '{bundle_path}'.\n")

if __name__ == '__main__':
    train_and_evaluate_models()
