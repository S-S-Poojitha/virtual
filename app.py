from flask import Flask, render_template, request, redirect, url_for
from transcript_extractor import process_youtube_video

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        youtube_url = request.form["youtube_url"]
        if not youtube_url:
            return render_template("index.html", error="Please enter a valid YouTube URL.")
        
        # Process the YouTube video
        result = process_youtube_video(youtube_url, num_sentences=3)
        
        if not result["success"]:
            return render_template("index.html", error=result["error"])
        
        return render_template("result.html", 
                               transcript=result["text"], 
                               summary=result["summary"])
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
