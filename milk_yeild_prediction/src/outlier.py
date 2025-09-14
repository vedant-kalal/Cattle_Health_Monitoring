import numpy as np

def cap_outliers(df, num_cols):
    df = df.copy()
    for col in num_cols:
        q1, q99 = df[col].quantile([0.01, 0.99])
        df[col] = np.clip(df[col], q1, q99)
    return df
