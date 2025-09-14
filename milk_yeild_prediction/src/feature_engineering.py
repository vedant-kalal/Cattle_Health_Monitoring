import numpy as np

def feature_engineering(df):
    df = df.copy()
    # Interaction feature: temperature × humidity (heat stress index)
    df["heat_index"] = (df["ambient_temp"] * df["humidity"]) / 100.0
    # Milk consistency ratio (doesn't use target directly)
    if "milk_yesterday" in df and "milk_7day_avg" in df:
        df["milk_ratio"] = df["milk_yesterday"] / (df["milk_7day_avg"] + 1e-5)
    return df
