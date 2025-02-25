# Improved YouTube transcript fetcher and summarizer
import requests
import json
import re
from flask import Flask, render_template, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import random
import string
import time

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

def generate_visitor_id():
    """Generate a random visitor ID similar to what YouTube uses."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(11))

def get_transcript(video_id):
    """Get the transcript for a YouTube video using direct API request with improved parameters."""
    try:
        url = f"https://www.youtube.com/youtubei/v1/get_transcript"
        
        # Parameters needed for the request
        params = {
            "key": "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"  # Public YouTube API key
        }
        
        visitor_id = generate_visitor_id()
        
        # Request headers with necessary values
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Origin": "https://www.youtube.com",
            "Referer": f"https://www.youtube.com/watch?v={video_id}",
            "X-YouTube-Client-Name": "1",
            "X-YouTube-Client-Version": "2.20250222.10.01",
            "X-Goog-Visitor-Id": visitor_id
        }
        
        # Request payload based on YouTube's current API format
        payload = {
            "context": {
                "client": {
                    "hl": "en",
                    "gl": "US",
                    "clientName": "WEB",
                    "clientVersion": "2.20250222.10.01",
                    "originalUrl": f"https://www.youtube.com/watch?v={video_id}",
                    "visitorData": visitor_id
                }
            },
            "params": video_id
        }
        
        # Make the request
        response = requests.post(url, params=params, headers=headers, json=payload)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return f"Error: Failed to retrieve transcript. Status code: {response.status_code}"
        
        # Parse the response
        data = response.json()
        
        # Extract transcript segments - YouTube API returns different structures
        transcript_text = ""
        
        # Try to extract from different possible response formats
        try:
            # Format 1: Newer YouTube API
            actions = data.get('actions', [])
            for action in actions:
                if 'updateEngagementPanelAction' in action:
                    panel = action['updateEngagementPanelAction']
                    content = panel.get('content', {})
                    if 'transcriptRenderer' in content:
                        transcript = content['transcriptRenderer']
                        body = transcript.get('body', {})
                        cue_groups = body.get('transcriptBodyRenderer', {}).get('cueGroups', [])
                        
                        for group in cue_groups:
                            cues = group.get('transcriptCueGroupRenderer', {}).get('cues', [])
                            for cue in cues:
                                text = cue.get('transcriptCueRenderer', {}).get('cue', {}).get('simpleText', '')
                                transcript_text += text + " "
        except Exception as e:
            print(f"Format 1 parsing error: {e}")
            
        try:
            # Format 2: Alternative structure
            if not transcript_text and 'captions' in data:
                captions = data['captions']
                renderer = captions.get('playerCaptionsTracklistRenderer', {})
                tracks = renderer.get('captionTracks', [])
                
                if tracks:
                    base_url = tracks[0].get('baseUrl')
                    if base_url:
                        # Fetch the transcript content from the base URL
                        transcript_response = requests.get(base_url)
                        if transcript_response.status_code == 200:
                            # Parse the transcript data (format depends on YouTube's response)
                            for line in transcript_response.text.split('\n'):
                                if not line.startswith('<') and line.strip():
                                    transcript_text += line.strip() + " "
        except Exception as e:
            print(f"Format 2 parsing error: {e}")
            
        # If we still don't have a transcript, try a third method using pytube
        if not transcript_text:
            try:
                from pytube import YouTube
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                caption_tracks = yt.captions
                
                if caption_tracks:
                    # Try to get English captions, or just use the first available
                    caption = None
                    for track in caption_tracks:
                        if 'en' in track.code.lower():
                            caption = track
                            break
                    
                    # If no English found, use the first one
                    if not caption and len(caption_tracks) > 0:
                        caption = list(caption_tracks)[0]
                    
                    if caption:
                        transcript_text = caption.generate_srt_captions()
                        
                        # Clean up SRT format
                        cleaned_text = ""
                        for line in transcript_text.split('\n'):
                            if not re.match(r'^\d+$', line) and not re.match(r'^\d\d:\d\d:\d\d', line) and line.strip():
                                cleaned_text += line + " "
                        transcript_text = cleaned_text
            except Exception as e:
                print(f"Pytube fallback error: {e}")
        
        if not transcript_text:
            return "Error: Transcript not available for this video or format not recognized."
        
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
