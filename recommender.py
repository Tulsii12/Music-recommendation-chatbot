import pandas as pd

df = pd.read_csv("tracks_clean.csv")

def recommend(mood=None, genre=None, n=5):
    result = df
    if mood:
        result = result[result["mood"] == mood]
    if genre:
        filtered = result[result["track_genre"] == genre]
        # if that genre has no tracks for this mood, keep the mood-only results
        if not filtered.empty:
            result = filtered
    if result.empty:
        return []
    return result.sample(min(n, len(result))).to_dict("records")