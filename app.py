# Direct YouTube transcript fetcher and summarizer
import requests
import json
import re
from flask import Flask, render_template, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

app = Flask(__name__)

def get_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    if 'youtu.be' in youtube_url:
        return youtube_url.split('/')[-1].split('?')[0]
    elif 'youtube.com/watch' in youtube_url:
        match = re.search(r'v=([^&]+)', youtube_url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Get the transcript for a YouTube video using direct API request."""
    try:
        url = "https://www.youtube.com/youtubei/v1/get_transcript"
        
        # Parameters needed for the request
        params = {
            "prettyPrint": "false"
        }
        
        # Request headers with minimal necessary values
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Origin": "https://www.youtube.com",
            "Referer": f"https://www.youtube.com/watch?v={video_id}"
        }
        
        # Request payload based on your provided example
        payload = {
            "context": {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": "2.20250222.10.01"
                }
            },
            "params": {
                "videoId": video_id
            }
        }
        
        # Make the request
        response = requests.post(url, params=params, headers=headers, json=payload)
        
        # Check if request was successful
        if response.status_code != 200:
            return f"Error: Failed to retrieve transcript. Status code: {response.status_code}"
        
        # Parse the response
        data = response.json()
        
        # Extract transcript segments - handle different response formats
        transcript_text = ""
        
        # Navigate through the JSON structure to find transcript segments
        if 'actions' in data:
            for action in data['actions']:
                if 'updateEngagementPanelAction' in action:
                    content = action.get('updateEngagementPanelAction', {}).get('content', {})
                    if 'transcriptRenderer' in content:
                        segments = content['transcriptRenderer'].get('body', {}).get('transcriptBodyRenderer', {}).get('cueGroups', [])
                        
                        for cue_group in segments:
                            for cue in cue_group.get('transcriptCueGroupRenderer', {}).get('cues', []):
                                cue_renderer = cue.get('transcriptCueRenderer', {})
                                text = cue_renderer.get('cue', {}).get('simpleText', '')
                                transcript_text += text + " "
        
        if not transcript_text:
            return "Error: Transcript format not recognized or transcript not available."
        
        return transcript_text
        
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return f"Error retrieving transcript: {str(e)}"

def summarize_text(text, num_sentences=5):
    """Summarize text using TextRank."""
    try:
        if not text.strip() or text.startswith("Error"):
            return "Error during summarization: Empty or invalid transcript."

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)

        if not summary:
            return "Error during summarization: No summary generated."

        return ' '.join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Error during summarization: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        youtube_url = data.get('youtube_url', '')

        print(f"Received URL: {youtube_url}")

        video_id = get_video_id(youtube_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'})

        transcript = get_transcript(video_id)
        if transcript.startswith('Error'):
            return jsonify({'error': transcript})

        summary = summarize_text(transcript)
        if "Error" in summary:
            return jsonify({'error': summary})

        return jsonify({
            'video_id': video_id,
            'summary': summary
        })

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
