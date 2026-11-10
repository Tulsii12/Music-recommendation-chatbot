import json, os, time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv
from recommender import recommend

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)
MODEL = "gemini-3.8-flash"   # if you get a 404, check the current model name in AI Studio

EXTRACT_PROMPT = """Extract the user's music intent. Reply ONLY with JSON:
{"mood": one of ["happy","sad","chill","angry",null],
 "genre": one of ["pop","rock","hip-hop","classical","jazz","edm","acoustic","indie",null]}
Map feelings to the closest mood (e.g. stressed/relaxed -> chill, heartbroken/lonely -> sad,
excited/party -> happy, frustrated -> angry). Use null if not mentioned."""

REPLY_PROMPT = """You are a friendly music buddy. Recommend ONLY the tracks provided below.
Do not invent songs. Write a short, warm reply (max 4 sentences) that acknowledges the
user's mood, then list the tracks as 'Title - Artist'. If the list is empty, ask the user
to tell you their mood or a genre."""


def call_gemini(prompt, system, temperature, json_mode=False):
    """Call Gemini with up to 3 attempts, so temporary 503 errors don't break the chat."""
    for attempt in range(3):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json" if json_mode else "text/plain",
            )
            return client.models.generate_content(model=MODEL, contents=prompt, config=config)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed:", e)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))   # wait 2s, then 4s


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    history = request.json.get("history", [])

    try:
        # 1) Understand intent (mood + genre) from the user's message
        r = call_gemini(user_msg, EXTRACT_PROMPT, 0, json_mode=True)
        intent = json.loads(r.text)

        # 2) Retrieve real tracks from your dataset
        tracks = recommend(intent.get("mood"), intent.get("genre"))
        track_text = "\n".join(f"{t['track_name']} - {t['artists']}" for t in tracks)

        # 3) Generate a grounded reply, with recent chat history for context
        convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
        prompt = (
            f"Conversation so far:\n{convo}\n\n"
            f"Tracks you may recommend:\n{track_text}\n\n"
            f"User: {user_msg}"
        )
        reply = call_gemini(prompt, REPLY_PROMPT, 0.7)
        return jsonify({"reply": reply.text})

    except Exception as e:
        print("ERROR:", e)
        # Fallback: still show songs from the dataset if the AI is unavailable
        tracks = recommend(None, None)
        text = "\n".join(f"{t['track_name']} - {t['artists']}" for t in tracks)
        return jsonify({
            "reply": "My AI is busy right now, but here are some tracks you might like:\n" + text
        })


if __name__ == "__main__":
    app.run(debug=True)