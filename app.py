from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import re

# Initialize Flask app
app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key="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")

def get_video_id(youtube_url):
    """Extract video ID from a YouTube URL."""
    match = re.search(r"v=([^&]+)", youtube_url)
    return match.group(1) if match else None

def fetch_transcript(video_id):
    """Retrieve transcript for a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"]).fetch()
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def summarize_with_groq(text):
    """Summarize text using Groq's LLaMA-3.3-70B model."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Summarize the following text into bullet points:\n{text}"}],
            temperature=0.7,
            max_completion_tokens=500,
            top_p=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during summarization: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        youtube_url = request.form.get("youtube_url")
        video_id = get_video_id(youtube_url)

        if not video_id:
            return render_template("index.html", error="Invalid YouTube URL.")

        transcript = fetch_transcript(video_id)
        if transcript.startswith("Error"):
            return render_template("index.html", error=transcript)

        summary = summarize_with_groq(transcript)

        return render_template("result.html", transcript=transcript, summary=summary)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
