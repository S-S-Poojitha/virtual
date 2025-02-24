import os
import streamlit as st
import tempfile
from TTS.api import TTS
import io
import numpy as np
import time
from pydub import AudioSegment
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import base64

# Set page configuration
st.set_page_config(
    page_title="Voice Cloning TTS",
    page_icon="🎙️",
    layout="wide"
)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Function to detect emotion from text
def detect_emotion(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return "happy"
    elif sentiment_scores['compound'] <= -0.05:
        return "sad"
    else:
        return "neutral"

# Function to modify audio based on emotion
def apply_emotion_effects(audio_data, emotion="neutral"):
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    
    if emotion == "happy":
        # For happy: increase speed, raise pitch, increase volume slightly
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 1.05)
        })
        audio = audio + 2  # Increase volume
        
    elif emotion == "sad":
        # For sad: lower pitch, decrease volume slightly
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 0.95)
        })
        audio = audio - 1  # Decrease volume
    
    # Export the modified audio
    output = io.BytesIO()
    audio.export(output, format="wav")
    return output.getvalue()

# Function to get TTS parameters based on emotion
def get_tts_params(emotion):
    params = {
        "happy": {"speed": 1.0},
        "sad": {"speed": 0.8},
        "neutral": {"speed": 0.85}
    }
    return params.get(emotion, {"speed": 0.85})

# Load TTS model
@st.cache_resource
def load_tts_model():
    os.environ["COQUI_TOS_AGREED"] = "1"
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Set up app UI
st.title("🎙️ Voice Cloning Text-to-Speech")
st.markdown("Upload or record your voice, and let AI clone it to speak Hindi text with appropriate emotions!")

# Initialize session state
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None

# Audio recorder
def start_recording():
    st.session_state.is_recording = True
    st.session_state.start_time = time.time()

def stop_recording():
    st.session_state.is_recording = False

# Set up columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Voice Input")
    # Upload option
    uploaded_file = st.file_uploader("Upload your voice sample (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])
    
    # Recording option
    st.write("Or record your voice:")
    if not st.session_state.is_recording:
        st.button("Start Recording", on_click=start_recording, type="primary")
    else:
        elapsed_time = time.time() - st.session_state.start_time
        st.write(f"Recording... {elapsed_time:.1f}s")
        if st.button("Stop Recording", type="secondary"):
            stop_recording()
            # Here we'd actually save the recording, but Streamlit doesn't have native audio recording
            # In a real implementation, you'd need a JavaScript component or external solution
            st.info("Note: In this demo, we're simulating recording. In a production app, you would need to integrate a JavaScript component for actual browser recording.")
            # Simulating a recorded file by using the uploaded file or a placeholder
            if uploaded_file:
                st.session_state.recorded_audio = uploaded_file.getvalue()
                st.success("Recording saved (simulated with uploaded file)")
            else:
                st.warning("Please upload a sample file to simulate recording")

    # Display the audio source
    if uploaded_file is not None:
        st.session_state.audio_file_path = uploaded_file
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        st.success("Voice sample uploaded!")
    elif st.session_state.recorded_audio is not None:
        st.audio(st.session_state.recorded_audio, format="audio/wav")

with col2:
    st.subheader("Text Input")
    
    # Language selection
    language = st.selectbox("Language", 
                           options=["hi", "en", "es", "fr", "de", "zh-cn", "ja", "ko"],
                           index=0)
    
    # Emotion detection toggle
    auto_emotion = st.checkbox("Automatically detect emotion from text", value=True)
    
    if not auto_emotion:
        emotion = st.radio("Select emotion", options=["neutral", "happy", "sad"], horizontal=True)
    
    # Default Hindi text
    default_text = "बहुत समय पहले, एक पूजिता नाम की लड़की थी। वह बहुत बुद्धिमान और साहसी थी।"
    
    # Text input
    text_input = st.text_area("Enter text to convert to speech", 
                             value=default_text,
                             height=150)
    
    # Speed control
    speed = st.slider("Speech Speed", min_value=0.7, max_value=1.3, value=0.85, step=0.05)

# Generate button
if st.button("Generate Speech", type="primary", use_container_width=True, disabled=not (uploaded_file is not None or st.session_state.recorded_audio is not None)):
    if uploaded_file is not None or st.session_state.recorded_audio is not None:
        # Load TTS model
        with st.spinner("Loading TTS model..."):
            tts = load_tts_model()
        
        # Get audio source
        audio_source = uploaded_file if uploaded_file is not None else st.session_state.recorded_audio
        
        # Save the uploaded/recorded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            if isinstance(audio_source, bytes):
                tmp.write(audio_source)
            else:
                tmp.write(audio_source.getvalue())
            temp_path = tmp.name
        
        try:
            # Determine emotion if auto-detection is enabled
            if auto_emotion:
                with st.spinner("Detecting emotion from text..."):
                    # For Hindi, we'd ideally have a Hindi sentiment analyzer
                    # This is a simplification using English sentiment analysis
                    emotion = detect_emotion(text_input)
                    st.info(f"Detected emotion: {emotion}")
            
            # Get parameters based on emotion
            tts_params = get_tts_params(emotion if auto_emotion else locals().get('emotion', 'neutral'))
            
            # Generate speech
            with st.spinner("Generating speech with your voice..."):
                output_buffer = io.BytesIO()
                tts.tts_to_file(
                    text=text_input,
                    file_path=output_buffer,
                    speaker_wav=temp_path,
                    language=language,
                    speed=speed if not auto_emotion else tts_params["speed"]
                )
                
                # Apply emotion effects
                final_audio = apply_emotion_effects(
                    output_buffer.getvalue(), 
                    emotion if auto_emotion else locals().get('emotion', 'neutral')
                )
            
            # Display audio player
            st.subheader("Generated Speech")
            st.audio(final_audio, format="audio/wav")
            
            # Download button
            st.download_button(
                label="Download audio",
                data=final_audio,
                file_name=f"voice_clone_{emotion}.wav",
                mime="audio/wav"
            )
            
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    else:
        st.warning("Please upload or record a voice sample first")

# Add information
with st.expander("Tips for better voice quality"):
    st.markdown("""
    - Use a clear voice recording of at least 6-10 seconds
    - Record in a quiet environment with minimal background noise
    - Speak naturally and expressively in your reference recording
    - For Hindi text, try to use proper punctuation for natural pauses
    - Adjust the speed slider for more natural-sounding output
    """)

# Add info about recording limitation
st.sidebar.info("""
**Note on Recording**: Streamlit doesn't natively support audio recording.
For a production application, you would need to integrate a JavaScript component.
In this demo, the recording function is simulated - when you click "Start Recording",
it will use your uploaded file as a substitute for an actual recording.
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("#### About")
st.sidebar.info("This app uses Coqui XTTS v2 for multilingual text-to-speech with voice cloning capabilities and emotion detection.")
