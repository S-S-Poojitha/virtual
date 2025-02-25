# app.py
from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
import re
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
    """Get the transcript for a YouTube video."""
    try:
        transcript_list = None
        # Use list_transcripts to get available transcripts
        transcript_info = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Prefer manually created transcript
        if transcript_info.find_manually_created_transcript(['en']):
            transcript_list = transcript_info.find_manually_created_transcript(['en']).fetch()
        # Fallback to auto-generated transcript
        elif transcript_info.find_generated_transcript(['en']):
            transcript_list = transcript_info.find_generated_transcript(['en']).fetch()
        else:
            return "Error: No available transcript in English."

    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcript found for this video."
    except CouldNotRetrieveTranscript as e:
        print(f"Error retrieving transcript: {e}")  # Debug log
        return f"Error retrieving transcript: {str(e)}"
    except Exception as e:
        print(f"Unexpected error retrieving transcript: {e}")  # Debug log
        return f"Error retrieving transcript: {str(e)}"

    transcript = ' '.join([item['text'] for item in transcript_list])
    print(f"Transcript length: {len(transcript)} characters")  # Debug log
    return transcript

def summarize_text(text, num_sentences=5):
    """Summarize text using TextRank."""
    try:
        if not text.strip():
            print("Transcript is empty.")  # Debug log
            return "Error during summarization: Empty transcript."

        print(f"Summarizing text of length: {len(text)}")  # Debug log
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)

        if not summary:
            print("No summary generated.")  # Debug log
            return "Error during summarization: No summary generated."

        return ' '.join(str(sentence) for sentence in summary)
    except Exception as e:
        print(f"Error during summarization: {e}")  # Debug log
        return f"Error during summarization: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        youtube_url = data.get('youtube_url', '')

        print(f"Received URL: {youtube_url}")  # Debug log

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
        print(f"Unexpected error: {e}")  # Debug log
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
