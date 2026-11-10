import pandas as pd

# 1. Load the raw data
df = pd.read_csv("data.csv")
print("Original size:", df.shape)

# 2. Keep only the columns we need
df = df[["track_name", "artists", "track_genre", "valence", "energy", "danceability", "tempo"]]

# 3. Clean: remove empty rows and duplicate songs
df = df.dropna().drop_duplicates(subset=["track_name", "artists"])

# 4. Keep a few genres, max 200 songs each (~1500 tracks)
genres = ["pop", "rock", "hip-hop", "classical", "jazz", "edm", "acoustic", "indie"]
df = df[df["track_genre"].isin(genres)]
df = df.groupby("track_genre").head(200).reset_index(drop=True)

# 5. Label each song's mood using valence (positivity) and energy
def label_mood(row):
    v, e = row["valence"], row["energy"]
    if v > 0.6 and e > 0.6: return "happy"
    if v > 0.6 and e <= 0.6: return "chill"
    if v <= 0.4 and e > 0.6: return "angry"
    if v <= 0.4 and e <= 0.4: return "sad"
    return "neutral"

df["mood"] = df.apply(label_mood, axis=1)
df = df[df["mood"] != "neutral"]

# 6. Dataset analysis: check balance
print("\nMood counts:\n", df["mood"].value_counts())
print("\nGenre counts:\n", df["track_genre"].value_counts())

# 7. Save the cleaned data
df.to_csv("tracks_clean.csv", index=False)
print("\nSaved", len(df), "tracks to tracks_clean.csv")