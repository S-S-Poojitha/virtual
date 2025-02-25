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
import os
from PIL import Image
import cv2
import pytesseract

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

def text_extractor(image_path):
    """Extract text from an image using pytesseract OCR."""
    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
            
        # Read image using PIL
        image = Image.open(image_path)
        
        # Extract text
        text = pytesseract.image_to_string(image, lang='eng')
        
        if not text.strip():
            # Try with OpenCV preprocessing if initial attempt yields no results
            img = cv2.imread(image_path)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply threshold to get image with only black and white
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            # Save preprocessed image
            preprocessed_path = f"{os.path.splitext(image_path)[0]}_preprocessed.jpg"
            cv2.imwrite(preprocessed_path, thresh)
            # Try OCR again
            text = pytesseract.image_to_string(Image.open(preprocessed_path), lang='eng')
            # Clean up
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
                
        return text.strip()
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    """Extract frames from a video file at specified frame rate."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Error: Could not open video file {video_path}"
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate)
        
        frames_info = []
        count = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Calculate timestamp
                timestamp = count / fps
                frames_info.append({
                    'path': frame_path,
                    'timestamp': timestamp
                })
                
                frame_count += 1
                
            count += 1
            
        cap.release()
        return frames_info
    except Exception as e:
        return f"Error extracting frames from video: {str(e)}"

def create_srt_file(subtitles, output_file):
    """Create an SRT subtitle file from extracted text and timestamps."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for index, (start_time, end_time, text) in enumerate(subtitles, start=1):
                f.write(str(index) + '\n')
                f.write("{:02d}:{:02d}:{:02d},{:03d} --> {:02d}:{:02d}:{:02d},{:03d}\n".format(
                    int(start_time / 3600),
                    int((start_time % 3600) / 60),
                    int(start_time % 60),
                    int((start_time % 1) * 1000),
                    int(end_time / 3600),
                    int((end_time % 3600) / 60),
                    int(end_time % 60),
                    int((end_time % 1) * 1000)
                ))
                f.write(text + '\n\n')
        return f"SRT file created successfully at {output_file}"
    except Exception as e:
        return f"Error creating SRT file: {str(e)}"

def video_to_transcript(video_path, output_srt=None):
    """Process a video file to extract transcript using OCR on frames."""
    try:
        # Create temporary directory for frames
        temp_dir = "temp_frames"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Extract frames
        frames_info = extract_frames_from_video(video_path, temp_dir)
        if isinstance(frames_info, str) and frames_info.startswith("Error"):
            return frames_info
            
        # Process each frame with OCR
        subtitles = []
        prev_text = ""
        
        for i in range(len(frames_info)):
            frame = frames_info[i]
            text = text_extractor(frame['path'])
            
            # Avoid duplicate text
            if text != prev_text and text and not text.startswith("Error"):
                end_time = frames_info[i+1]['timestamp'] if i+1 < len(frames_info) else frame['timestamp'] + 1.0
                subtitles.append((frame['timestamp'], end_time, text))
                prev_text = text
                
        # Clean up frames
        for frame in frames_info:
            if os.path.exists(frame['path']):
                os.remove(frame['path'])
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            
        # Create SRT file if requested
        transcript_text = ""
        if output_srt:
            result = create_srt_file(subtitles, output_srt)
            if result.startswith("Error"):
                return result
                
        # Create plain text transcript
        for _, _, text in subtitles:
            transcript_text += text + " "
            
        return transcript_text
    except Exception as e:
        return f"Error processing video to transcript: {str(e)}"

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
        use_ocr = data.get('use_ocr', False)
        video_file = data.get('video_file', '')
        image_file = data.get('image_file', '')

        # Process based on input type
        transcript = ""
        
        if youtube_url:
            print(f"Processing YouTube URL: {youtube_url}")
            video_id = get_video_id(youtube_url)
            if not video_id:
                return jsonify({'error': 'Invalid YouTube URL'})
                
            transcript = get_transcript(video_id)
            
        elif video_file and use_ocr:
            print(f"Processing video file with OCR: {video_file}")
            if not os.path.exists(video_file):
                return jsonify({'error': 'Video file not found'})
                
            transcript = video_to_transcript(video_file)
            
        elif image_file and use_ocr:
            print(f"Processing image file with OCR: {image_file}")
            if not os.path.exists(image_file):
                return jsonify({'error': 'Image file not found'})
                
            transcript = text_extractor(image_file)
            
        else:
            return jsonify({'error': 'No valid input provided. Please provide a YouTube URL, video file, or image file.'})

        if transcript.startswith('Error'):
            return jsonify({'error': transcript})

        summary = summarize_text(transcript)
        if "Error" in summary:
            return jsonify({'error': summary})

        return jsonify({
            'transcript': transcript,
            'summary': summary
        })

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

@app.route('/ocr', methods=['POST'])
def process_ocr():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        # Save the uploaded file
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        # Process based on file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Process image with OCR
            text = text_extractor(file_path)
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Process video with OCR
            text = video_to_transcript(file_path)
        else:
            return jsonify({'error': 'Unsupported file type'})
            
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
            
        if text.startswith('Error'):
            return jsonify({'error': text})
            
        # Generate summary
        summary = summarize_text(text)
        
        return jsonify({
            'text': text,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing OCR: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
