import pandas as pd

df = pd.read_csv("tracks_clean.csv")

# Ideal (valence, energy) for each mood
TARGETS = {
    "happy": (0.8, 0.8),
    "chill": (0.65, 0.3),
    "angry": (0.25, 0.8),
    "sad":   (0.2, 0.3),
}

def clean_artists(text):
    # keep at most 2 artists, separated by commas
    return ", ".join(str(text).split(";")[:2])

def recommend(mood=None, genre=None, n=5):
    result = df
    if mood:
        result = result[result["mood"] == mood]
    if genre:
        filtered = result[result["track_genre"] == genre]
        if not filtered.empty:        # fall back to mood-only if none match
            result = filtered
    if result.empty:
        return []

    # Rank by closeness to the mood's ideal profile, then sample for variety
    if mood in TARGETS:
        v, e = TARGETS[mood]
        result = result.assign(dist=(result["valence"] - v) ** 2 + (result["energy"] - e) ** 2)
        result = result.nsmallest(max(n * 3, n), "dist")

    picks = result.sample(min(n, len(result))).to_dict("records")
    for p in picks:
        p["artists"] = clean_artists(p["artists"])
    return picks