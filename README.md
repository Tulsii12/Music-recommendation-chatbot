# music-recommendation-chatbot
Music recommendation generative AI and prompt engineering project.
# Music Recommendation Chatbot

A conversational chatbot that recommends songs based on your mood and favourite genre.
Tell it "I'm stressed, something relaxing" and it replies with real tracks from a dataset.

## How it works

1. **Understand the request:** the Gemini API reads the message and extracts a mood
   (happy, sad, chill, angry) and an optional genre.
2. **Pick real songs:** a pandas recommender filters a cleaned dataset of 10000+ tracks
   by mood and genre. Mood is derived from the audio features `valence` and `energy`.
3. **Write the reply:** Gemini turns the chosen tracks into a friendly answer, using
   recent chat history for context. It can only recommend the tracks it is given,
   so it never invents songs.

If the AI is unavailable, the app retries and then falls back to showing songs
straight from the dataset.

## Tech stack

Python, Flask, pandas, Gemini API (google-genai), HTML, CSS, JavaScript

## Run it locally

1. Clone the repo and open the folder.
2. Create and activate a virtual environment:
   `python -m venv venv` then `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Get a free API key from https://aistudio.google.com/apikey and create a `.env` file:
   `GEMINI_API_KEY=your_key_here`
5. Run `python app.py` and open http://127.0.0.1:5000

To rebuild the dataset from scratch, download the "Spotify Tracks Dataset" from Kaggle,
save it as `data.csv`, and run `python prepare_data.py`.

## Screenshot

![Chat demo](C:\Users\HP\Downloads\Music_recommendation_chatbot\Music-recommendation-chatbot\chat.png)

## What I learned

- Cleaning data and creating mood labels from audio features improved recommendation quality.
- Grounding the AI in a real dataset prevents made-up songs.
- Handling API errors (retries and a fallback) makes the app more reliable.