import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath, split=False):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    if split:
        X = df.drop("diabetes", axis=1)
        y = df["diabetes"]

        # 80/10/10 split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.10, stratify=y, random_state=42
        )

        val_ratio = 0.10 / 0.90  
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    return df
