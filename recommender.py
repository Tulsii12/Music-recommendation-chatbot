import pandas as pd

df = pd.read_csv("tracks_clean.csv")

# Ideal (valence, energy) for each mood. Songs closest to these are ranked first.
TARGETS = {
    "happy": (0.8, 0.8),
    "chill": (0.65, 0.3),
    "angry": (0.25, 0.8),
    "sad":   (0.2, 0.3),
}


def clean_artists(text):
    """The dataset joins artists with ';'. Keep at most 2, separated by commas."""
    return ", ".join(str(text).split(";")[:2])


def recommend(mood=None, genre=None, n=5):
    result = df

    # 1) Filter by mood
    if mood:
        result = result[result["mood"] == mood]

    # 2) Filter by genre; if no song matches both, keep the mood-only results
    if genre:
        filtered = result[result["track_genre"] == genre]
        if not filtered.empty:
            result = filtered
    else:
        # No genre asked for: skip classical so it doesn't dominate generic requests
        non_classical = result[result["track_genre"] != "classical"]
        if not non_classical.empty:
            result = non_classical

    if result.empty:
        return []

    # 3) Rank by closeness to the mood's ideal profile, keep the best few
    if mood in TARGETS:
        v, e = TARGETS[mood]
        result = result.assign(
            dist=(result["valence"] - v) ** 2 + (result["energy"] - e) ** 2
        )
        result = result.nsmallest(max(n * 3, n), "dist")

    # 4) Sample from the best matches so repeat requests give different songs
    picks = result.sample(min(n, len(result))).to_dict("records")
    for p in picks:
        p["artists"] = clean_artists(p["artists"])
    return picks