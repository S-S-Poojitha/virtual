import os
import re
import pandas as pd
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    if 'youtu.be' in youtube_url:
        return youtube_url.split('/')[-1].split('?')[0]
    elif 'youtube.com/watch' in youtube_url:
        match = re.search(r'v=([^&]+)', youtube_url)
        if match:
            return match.group(1)
    return None

def get_transcript_selenium(youtube_url, headless=True):
    try:
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Set a unique user data directory to avoid conflicts
        options.add_argument(f'--user-data-dir=/tmp/chrome-user-data-{os.getpid()}')
        
        # Initialize WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        driver.get(youtube_url)

        # Continue with the existing code...
        
        driver.quit()
        return {'success': True, 'text': "Sample transcript"}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def summarize_text(text, num_sentences=5):
    """Summarize text using TextRank algorithm."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def process_youtube_video(youtube_url, num_sentences=5, headless=True):
    """Main function to process a YouTube video."""
    video_id = get_video_id(youtube_url)
    if not video_id:
        return {"success": False, "error": "Invalid YouTube URL"}

    transcript_result = get_transcript_selenium(youtube_url, headless=headless)
    if not transcript_result["success"]:
        return transcript_result

    transcript_result["summary"] = summarize_text(transcript_result["text"], num_sentences)
    return transcript_result
