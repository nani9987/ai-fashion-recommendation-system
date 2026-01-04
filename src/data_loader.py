import pandas as pd
from src.config import CSV_PATH

def load_dataset():
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    return df

def apply_filters(df, color="All", season="All", usage="All"):
    filtered = df.copy()

    if color != "All":
        filtered = filtered[filtered["baseColour"] == color]

    if season != "All":
        filtered = filtered[filtered["season"] == season]

    if usage != "All":
        filtered = filtered[filtered["usage"] == usage]

    return filtered
