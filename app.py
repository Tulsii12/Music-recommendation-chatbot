import json, os
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv
from recommender import recommend

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)
MODEL = "gemini-2.5-flash"   # if this name errors, check the current model list in AI Studio

EXTRACT_PROMPT = """Extract the user's music intent. Reply ONLY with JSON:
{"mood": one of ["happy","sad","chill","angry",null],
 "genre": one of ["pop","rock","hip-hop","classical","jazz","edm","acoustic","indie",null]}
Map feelings to the closest mood (e.g. stressed/relaxed -> chill, heartbroken/lonely -> sad,
excited/party -> happy, frustrated -> angry). Use null if not mentioned."""

REPLY_PROMPT = """You are a friendly music buddy. Recommend ONLY the tracks provided below.
Do not invent songs. Write a short, warm reply (max 4 sentences) that acknowledges the
user's mood, then list the tracks as 'Title - Artist'. If the list is empty, ask the user
to tell you their mood or a genre."""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    history = request.json.get("history", [])

    try:
        # 1) Understand intent
        r = client.models.generate_content(
            model=MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACT_PROMPT,
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        intent = json.loads(r.text)

        # 2) Retrieve real tracks from your dataset
        tracks = recommend(intent.get("mood"), intent.get("genre"))
        track_text = "\n".join(f"{t['track_name']} - {t['artists']}" for t in tracks)

        # 3) Generate grounded reply, with recent chat history for context
        convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
        prompt = f"Conversation so far:\n{convo}\n\nTracks you may recommend:\n{track_text}\n\nUser: {user_msg}"

        reply = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=REPLY_PROMPT,
                temperature=0.7,
            ),
        )
        return jsonify({"reply": reply.text})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "Sorry, something went wrong. Please try again in a moment."})

if __name__ == "__main__":
    app.run(debug=True)