import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def load_and_clean_data(filepath, split=False, standardize=True):
    df = pd.read_csv(filepath)

    # Remove duplicates
    df = df.drop_duplicates()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    if not split:
        return df

    # Split features and target
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # First split: 80% (train + val) and 20% (test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Second split: from 80%, get 70% (train) and 10% (val)
    val_ratio = 0.10 / 0.80  # 12.5%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )

    # Standardize if required
    if standardize:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        # Convert to NumPy arrays (SMOTE requires this)
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values

    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return (
        X_train_resampled, X_val, X_test,
        y_train_resampled, y_val, y_test
    )
