import random

def get_recommendations(df, n=6):
    if len(df) == 0:
        return df
    return df.sample(min(n, len(df)))
