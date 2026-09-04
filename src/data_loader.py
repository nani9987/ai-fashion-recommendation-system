import pandas as pd
from pathlib import Path

def load_dataset():
    """Load fashion dataset with error handling."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "styles.csv"
    
    try:
        df = pd.read_csv(
            csv_path, 
            on_bad_lines="skip",
            encoding='utf-8'
        )
        
        # Normalize text columns to lowercase
        for col in ["baseColour", "season", "usage", "articleType"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV: {e}")

def apply_filters(df, color="All", season="All", usage="All"):
    """Filter dataset by attributes."""
    filtered = df.copy()
    
    if color != "All" and color.lower() in df["baseColour"].values:
        filtered = filtered[filtered["baseColour"] == color.lower()]
    
    if season != "All" and season.lower() in df["season"].values:
        filtered = filtered[filtered["season"] == season.lower()]
    
    if usage != "All" and usage.lower() in df["usage"].values:
        filtered = filtered[filtered["usage"] == usage.lower()]
    
    return filtered
