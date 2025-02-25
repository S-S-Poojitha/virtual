from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import re
from markupsafe import Markup

import markdown

# Initialize Flask app
app = Flask(__name__)

# Initialize Groq client (Replace with your actual API key)
client = Groq(api_key="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")

def get_video_id(youtube_url):
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"v=([^&]+)",                # Standard YouTube URL (https://www.youtube.com/watch?v=VIDEO_ID)
        r"youtu\.be/([^?]+)",         # Shortened YouTube URL (https://youtu.be/VIDEO_ID)
        r"youtube\.com/shorts/([^?]+)",  # Shorts URL (https://youtube.com/shorts/VIDEO_ID)
        r"youtube\.com/live/([^?]+)"  # Live Video URL (https://www.youtube.com/live/VIDEO_ID)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None
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

def markdown_filter(text):
    """Convert Markdown to HTML."""
    return Markup(markdown.markdown(text))

# Register Markdown filter in Jinja
app.jinja_env.filters["markdown"] = markdown_filter

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
