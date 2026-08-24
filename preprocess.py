import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CAFFEINE_MAP = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
MOOD_MAP = {'Anxious': 0, 'Sad': 1, 'Neutral': 2, 'Happy': 3}
INTERRUPTIONS_MAP = {'No': 0, 'Yes': 1}
TARGET_MAP = {'Poor': 0, 'Average': 1, 'Good': 2}
REVERSE_TARGET_MAP = {0: 'Poor', 1: 'Average', 2: 'Good'}

FEATURE_COLUMNS = [
    'sleep_duration',
    'bedtime_hour',
    'wakeup_hour',
    'is_late_bedtime',
    'caffeine_encoded',
    'exercise_duration',
    'screen_time',
    'stress_level',
    'mood_encoded',
    'interruptions_encoded'
]

def parse_time_to_hours(time_str: str) -> float:
    
    if isinstance(time_str, (str, pd.Series)):
        if isinstance(time_str, str):
            parts = time_str.split(':')
            return float(parts[0]) + float(parts[1]) / 60.0
        else:
            def _parse_val(v):
                if isinstance(v, str):
                    p = v.split(':')
                    return float(p[0]) + float(p[1]) / 60.0
                return float(v)
            return time_str.apply(_parse_val)
    elif hasattr(time_str, 'hour') and hasattr(time_str, 'minute'):
        return float(time_str.hour) + float(time_str.minute) / 60.0
    return float(time_str)

def calculate_overnight_duration(bedtime_h: float, wakeup_h: float) -> float:
    
    if wakeup_h >= bedtime_h:
        return wakeup_h - bedtime_h
    else:
        return (24.0 - bedtime_h) + wakeup_h

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    if 'bedtime_hour' not in df.columns:
        df['bedtime_hour'] = parse_time_to_hours(df['bedtime'])
    if 'wakeup_hour' not in df.columns:
        df['wakeup_hour'] = parse_time_to_hours(df['wakeup_time'])

    df['is_late_bedtime'] = df['bedtime_hour'].apply(
        lambda h: 1 if (h >= 23.0 or h < 4.0) else 0
    )

    if 'caffeine_intake' in df.columns and 'caffeine_encoded' not in df.columns:
        df['caffeine_encoded'] = df['caffeine_intake'].map(CAFFEINE_MAP).fillna(1).astype(int)

    if 'mood' in df.columns and 'mood_encoded' not in df.columns:
        df['mood_encoded'] = df['mood'].map(MOOD_MAP).fillna(2).astype(int)

    if 'sleep_interruptions' in df.columns and 'interruptions_encoded' not in df.columns:
        df['interruptions_encoded'] = df['sleep_interruptions'].map(INTERRUPTIONS_MAP).fillna(0).astype(int)

    if 'sleep_quality' in df.columns and df['sleep_quality'].dtype == object:
        df['target'] = df['sleep_quality'].map(TARGET_MAP)
    elif 'sleep_quality' in df.columns:
        df['target'] = df['sleep_quality']

    return df

def preprocess_dataframe(df: pd.DataFrame, scaler=None, is_training: bool = True):
    
    df_engineered = engineer_features(df)

    X = df_engineered[FEATURE_COLUMNS].copy()

    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when is_training=False")
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS, index=X.index)
    
    y = df_engineered['target'] if 'target' in df_engineered.columns else None

    return X_scaled_df, y, scaler

def preprocess_single_input(input_dict: dict, scaler) -> pd.DataFrame:
    
    raw_df = pd.DataFrame([input_dict])
    engineered_df = engineer_features(raw_df)
    
    X = engineered_df[FEATURE_COLUMNS].copy()
    X_scaled = scaler.transform(X)
    
    return pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
