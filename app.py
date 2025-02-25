from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import re

app = Flask(__name__)

def get_video_id(youtube_url):
    """Extract video ID from a YouTube URL."""
    match = re.search(r"v=([^&]+)", youtube_url)
    print(match.group(1))
    return match.group(1) if match else None

def fetch_transcript(video_id):
    """Fetch transcript using youtube_transcript_api."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def summarize_text(text, num_sentences=3):
    """Summarize text using Sumy (TextRank)."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
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

        summary = summarize_text(transcript, num_sentences=3)

        return render_template("result.html", transcript=transcript, summary=summary)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
