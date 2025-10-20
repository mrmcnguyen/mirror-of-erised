#!/usr/bin/env python3
"""
main.py - Mirror of Erised Interactive Voice Interface

This is the main entry point for the Mirror of Erised project.
It greets the user and then continuously listens for speech,
automatically stopping when the user stops talking.

Requirements:
  pip install pywhispercpp sounddevice numpy soundfile piper simpleaudio openai pygame pillow
  
Raspberry Pi 5 Setup:
  sudo apt update
  sudo apt install python3-pygame python3-pil
  # For better performance, consider:
  # sudo raspi-config -> Advanced Options -> Memory Split -> 128 (for GPU memory)
"""

import argparse
import numpy as np
import sounddevice as sd
import time
import threading
from pywhispercpp.model import Model
from piper import PiperVoice
import simpleaudio as sa
import wave
import os
from openai import OpenAI
import pygame
import random
from PIL import Image
import glob
import cv2
import mediapipe as mp
from deepface import DeepFace

class MirrorOfErised:
    def __init__(self, whisper_model="tiny-q5_1", piper_voice="en_US-lessac-medium.onnx", 
                 sample_rate=16000, silence_threshold=0.01, silence_duration=2.0, 
                 openai_api_key=None, openai_model="gpt-4o-mini", 
                 images_dir="emotion_images", display_width=1024, display_height=600, 
                 fullscreen=False, tts_speed=1.1):
        """
        Initialize the Mirror of Erised voice interface.
        
        Args:
            whisper_model: Whisper model name for speech recognition
            piper_voice: Piper voice model path for text-to-speech
            sample_rate: Audio sample rate (Hz)
            silence_threshold: Audio level threshold for silence detection
            silence_duration: Duration of silence before stopping recording (seconds)
            openai_api_key: OpenAI API key (if None, will try to get from environment)
            openai_model: OpenAI model to use (default: gpt-4o-mini for speed)
            images_dir: Directory containing emotion-based images
            display_width: Display width in pixels
            display_height: Display height in pixels
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.is_listening = False
        self.audio_buffer = []
        self.recording_thread = None
        
        # Display and emotion management
        self.images_dir = images_dir
        self.display_width = display_width
        self.display_height = display_height
        self.fullscreen = fullscreen
        self.current_emotion = None
        self.emotion_images = {}
        self.display_screen = None
        self.emotion_timer = None
        self.emotion_thread = None
        self.running = True
        
        # Camera and emotion detection
        self.camera = None
        self.emotion_detection_enabled = True
        self.last_emotion_detection = 0
        self.emotion_detection_interval = 5.0  # Detect emotion every 5 seconds
        self.emotion_confidence_threshold = 0.5
        self.frame_skip_counter = 0
        self.frame_skip_interval = 3  # Process every 3rd frame
        
        # Initialize Whisper model for speech recognition
        print(f"Loading Whisper model: {whisper_model}...")
        self.whisper_model = Model(whisper_model)
        
        # Initialize Piper voice for text-to-speech
        print(f"Loading Piper voice: {piper_voice}...")
        self.piper_voice = PiperVoice.load(piper_voice)
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
                self.openai_client = None
        
        self.openai_model = openai_model
        
        # Define the magical mirror prompt
        self.mirror_instructions = """
                You are a magical smart mirror inspired by the Mirror of Erised from the Harry Potter universe. 
                Your purpose is to react to the user's detected emotions by speaking to them in a whimsical, mystical,
                and engaging tone that feels true to the Harry Potter world.

                When responding:
                - Always keep responses under 25 words.
                - Always initiate or continue a two-way conversation. Each reply must include a follow-up question or
                  prompt that encourages the user to keep speaking with you.
                - Responses should feel enchanted, intriguing, and slightly mysterious, as though you are a wise but
                  playful magical artifact.
                - Vary your style so you don't sound repetitive or mundane. Break monotony by sometimes offering magical
                  choices, riddles, trivia, or whimsical curiosities from the wizarding world. Keep interactions fresh 
                  and unexpected. THIS POINT IS ESPECIALLY IMPORTANT
                - Adjust your tone to the user's detected emotion (e.g., comforting if sad, encouraging if happy, grounding
                  if anxious, fiery if angry).
                - You may reference magical concepts, objects, or creatures from the Harry Potter world, but do not break 
                  character or explain your purpose as an AI.
                - Each response must create anticipation for the user's next reply
                - Never generate long explanations; keep answers short, magical, and conversational.
                - Use varied sentence structures
                - Create emotional hooks that make users want to share more
                - Don't repeat the same question types consecutively
                - Don't add any emoji's to your responses

                Your role is to sustain an ongoing dialogue with the user, blending emotional empathy with fantasy charm, 
                while keeping responses surprising, lively, and interactive.
                """
        
        # Initialize display system
        self.init_display()
        self.load_emotion_images()
        
        # Initialize camera for emotion detection
        self.init_camera()
        
        print("Mirror of Erised is ready...")
    
    def add_mystical_effects(self, audio, sample_rate):
        """
        Add mystical reverb and echo effects to create a Mirror of Erised atmosphere
        """
        # Create multiple echo delays for mystical effect
        echo_delays = [0.3, 0.6, 1.2]  # seconds
        echo_amplitudes = [0.1, 0.05, 0.02]  # decreasing volume
        
        # Start with original audio
        mystical_audio = audio.copy()
        
        # Add multiple echoes
        for delay, amplitude in zip(echo_delays, echo_amplitudes):
            delay_samples = int(delay * sample_rate)
            if delay_samples < len(audio):
                # Create delayed version
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * amplitude
                mystical_audio += delayed
        
        # Add simple reverb using multiple delayed copies with decay
        reverb_delays = [0.05, 0.1, 0.15]  # seconds
        reverb_amplitudes = [0.15, 0.1, 0.08]
        
        for delay, amplitude in zip(reverb_delays, reverb_amplitudes):
            delay_samples = int(delay * sample_rate)
            if delay_samples < len(mystical_audio):
                delayed = np.zeros_like(mystical_audio)
                delayed[delay_samples:] = mystical_audio[:-delay_samples] * amplitude
                mystical_audio += delayed
        
        # Add a subtle low-pass filter effect (simple moving average)
        # This simulates the sound traveling through magical glass
        filter_size = int(0.01 * sample_rate)  # 10ms filter
        if filter_size > 1:
            kernel = np.ones(filter_size) / filter_size
            mystical_audio = np.convolve(mystical_audio, kernel, mode='same')
        
        # Add a subtle tremolo effect for mystical vibration
        tremolo_rate = 4.5  # Hz
        tremolo_depth = 0.15
        t = np.arange(len(mystical_audio)) / sample_rate
        tremolo = 1 + tremolo_depth * np.sin(2 * np.pi * tremolo_rate * t)
        mystical_audio *= tremolo
        
        # Add a subtle chorus effect for more mystical depth
        chorus_delay = int(0.02 * sample_rate)  # 20ms delay
        chorus_depth = 0.1
        if chorus_delay < len(mystical_audio):
            chorus_signal = np.zeros_like(mystical_audio)
            chorus_signal[chorus_delay:] = mystical_audio[:-chorus_delay] * chorus_depth
            mystical_audio += chorus_signal
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mystical_audio))
        if max_val > 0:
            mystical_audio = mystical_audio / max_val * 0.8
        
        return mystical_audio
    
    def speak(self, text, speed_factor=1.3):
        """
        Convert text to speech with mystical Mirror of Erised effects
        
        Args:
            text: Text to speak
            speed_factor: Speed multiplier (1.0 = normal, 1.3 = 30% faster, 1.5 = 50% faster)
        """
        print(f"Mirror says: {text}")
        
        # Generate audio using Piper
        audio_chunks = list(self.piper_voice.synthesize(text))
        
        if audio_chunks and len(audio_chunks) > 0:
            # Get the sample rate from the first chunk
            sample_rate = audio_chunks[0].sample_rate
            
            # Extract audio arrays from each chunk and concatenate
            audio_arrays = [chunk.audio_float_array for chunk in audio_chunks]
            audio = np.concatenate(audio_arrays)
        else:
            print("No audio chunks generated!")
            return
        
        # Speed up the audio by resampling
        if speed_factor != 1.0:
            # Calculate new length after speeding up
            new_length = int(len(audio) / speed_factor)
            # Resample using linear interpolation
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
        
        # Apply mystical Mirror of Erised effects
        mystical_audio = self.add_mystical_effects(audio, sample_rate)
        
        # Convert to 16-bit PCM
        pcm_audio = (mystical_audio * 32767).astype(np.int16)
        
        # Play the mystical audio at the original sample rate
        # (the audio is already sped up, so we play at normal rate)
        play_obj = sa.play_buffer(pcm_audio, 1, 2, sample_rate)
        play_obj.wait_done()

    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for real-time audio input
        """
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_listening:
            # Convert to mono and store in buffer
            audio_data = np.squeeze(indata)
            self.audio_buffer.extend(audio_data)
    
    def detect_silence(self, audio_data, threshold=None):
        """
        Detect if audio contains silence based on RMS energy
        """
        if threshold is None:
            threshold = self.silence_threshold
        
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < threshold
    
    def record_with_voice_activity_detection(self):
        """
        Record audio with voice activity detection - stops when user stops talking
        """
        print("🔮 The Mirror is listening... (speak now)")
        
        self.is_listening = True
        self.audio_buffer = []
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.2)  # 200ms blocks
        )
        
        stream.start()
        
        # Monitor for silence
        silence_start = None
        last_audio_time = time.time()
        
        try:
            while self.is_listening:
                time.sleep(0.2)  # Check every 200ms
                
                if len(self.audio_buffer) > 0:
                    # Check recent audio for voice activity
                    recent_audio = np.array(self.audio_buffer[-int(self.sample_rate * 0.5):])  # Last 0.5 seconds
                    
                    if not self.detect_silence(recent_audio):
                        # Voice detected, reset silence timer
                        silence_start = None
                        last_audio_time = time.time()
                    else:
                        # Silence detected
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.silence_duration:
                            # Silence for too long, stop recording
                            print("🔮 The Mirror has heard enough...")
                            break
                    
                    # Prevent buffer from growing too large
                    if len(self.audio_buffer) > self.sample_rate * 30:  # 30 seconds max
                        self.audio_buffer = self.audio_buffer[-int(self.sample_rate * 10):]  # Keep last 10 seconds
        
        except KeyboardInterrupt:
            print("\n🔮 Recording interrupted...")
        
        finally:
            self.is_listening = False
            stream.stop()
            stream.close()
        
        # Convert buffer to numpy array
        if self.audio_buffer:
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            return audio_array
        else:
            return None
    
    def transcribe_audio(self, audio_array):
        """
        Transcribe audio using Whisper
        """
        if audio_array is None or len(audio_array) == 0:
            return ""
        
        print("🔮 The Mirror is processing your words...")
        segments = self.whisper_model.transcribe(audio_array)
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(texts)
    
    def greet_user(self):
        """
        Initial greeting from the Mirror of Erised
        """
        greeting = (
            "I am the Mirror of Ery-sed. I show not your face but your heart's desire. "
            "Speak to me, and I shall reveal what lies within your soul. "
        )
        self.speak(greeting)
    
    def init_display(self):
        """
        Initialize pygame display for image rendering on Raspberry Pi 5
        """
        try:
            # Initialize pygame with optimizations for Pi 5
            pygame.init()
            
            # Set environment variables for better Pi 5 performance
            os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
            os.environ['SDL_VIDEO_CENTERED'] = '0'
            
            # Choose display mode based on fullscreen setting
            if self.fullscreen:
                # Fullscreen mode for production
                self.display_screen = pygame.display.set_mode(
                    (self.display_width, self.display_height),
                    pygame.FULLSCREEN | pygame.DOUBLEBUF
                )
                pygame.mouse.set_visible(False)  # Hide cursor in fullscreen
                print(f"Display initialized in fullscreen mode: {self.display_width}x{self.display_height}")
                print("🔮 Production mode: Fullscreen display")
            else:
                # Windowed mode for development
                self.display_screen = pygame.display.set_mode(
                    (self.display_width, self.display_height),
                    pygame.DOUBLEBUF
                )
                pygame.mouse.set_visible(True)  # Keep cursor visible for development
                print(f"Display initialized in windowed mode: {self.display_width}x{self.display_height}")
                print("🔮 Development mode: Windowed display - you can still use the desktop")
            
            pygame.display.set_caption("Mirror of Erised - Press ESC or close window to exit")
            
            # Black background
            self.display_screen.fill((0, 0, 0))
            pygame.display.flip()
            
            print(f"🔮 Display ready for 10-inch LCD (1024x600)")
            
        except Exception as e:
            print(f"Display initialization failed: {e}")
            self.display_screen = None
    
    def load_emotion_images(self):
        """
        Load images for each emotion from the images directory
        Expected structure: emotion_images/happy/, emotion_images/sad/, etc.
        """
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
        for emotion in emotions:
            emotion_dir = os.path.join(self.images_dir, emotion)
            if os.path.exists(emotion_dir):
                # Load all image files from the emotion directory
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']:
                    image_files.extend(glob.glob(os.path.join(emotion_dir, ext)))
                    image_files.extend(glob.glob(os.path.join(emotion_dir, ext.upper())))
                
                if image_files:
                    self.emotion_images[emotion] = image_files
                    print(f"Loaded {len(image_files)} images for emotion: {emotion}")
                else:
                    print(f"No images found for emotion: {emotion}")
            else:
                print(f"Directory not found for emotion: {emotion}")
        
        # If no images loaded, create a default structure message
        if not self.emotion_images:
            print(f"No emotion images found. Please create directories like: {self.images_dir}/happy/, {self.images_dir}/sad/, etc.")
    
    def get_random_emotion(self):
        """
        Get a random emotion from available emotions
        """
        available_emotions = list(self.emotion_images.keys())
        if available_emotions:
            return random.choice(available_emotions)
        else:
            # Fallback to predefined emotions if no images loaded
            return random.choice(["happy", "sad", "angry", "anxious", "neutral"])
    
    def get_random_image_for_emotion(self, emotion):
        """
        Get a random image for the specified emotion
        """
        if emotion in self.emotion_images and self.emotion_images[emotion]:
            return random.choice(self.emotion_images[emotion])
        return None
    
    def display_image(self, image_path):
        """
        Display an image vertically on the 10-inch LCD screen (1024x600) with Pi 5 optimizations
        """
        if not self.display_screen or not image_path or not os.path.exists(image_path):
            return
        
        try:
            # Load and scale the image with optimizations for Pi 5
            image = Image.open(image_path)
            
            # Calculate scaling to fit 1024x600 screen in vertical orientation
            img_width, img_height = image.size
            screen_width, screen_height = self.display_width, self.display_height
            
            # For vertical display, we want the image to fit the full height (600px)
            # and scale the width proportionally, then center horizontally
            target_height = screen_height  # Use full screen height
            scale = target_height / img_height
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # If the scaled width is wider than screen, scale down to fit width
            if new_width > screen_width:
                scale = screen_width / img_width
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
            
            # Optimize for Pi 5: Use faster resampling for large images
            if scale < 0.5:  # If scaling down significantly
                resample_method = Image.Resampling.LANCZOS
            else:
                resample_method = Image.Resampling.BILINEAR  # Faster for smaller scaling
            
            # Resize image
            image = image.resize((new_width, new_height), resample_method)
            
            # Convert to RGB (required for pygame)
            image_rgb = image.convert('RGB')
            
            # Convert to pygame surface with optimizations
            image_pygame = pygame.image.fromstring(
                image_rgb.tobytes(), 
                image_rgb.size, 
                image_rgb.mode
            )
            
            # Clear screen with black background
            self.display_screen.fill((0, 0, 0))
            
            # Center the image vertically on 1024x600 screen
            x = (screen_width - new_width) // 2
            y = (screen_height - new_height) // 2
            
            # Display the image
            self.display_screen.blit(image_pygame, (x, y))
            pygame.display.flip()
            
            print(f"🔮 Displaying vertically: {os.path.basename(image_path)} ({new_width}x{new_height})")
            
        except Exception as e:
            print(f"Error displaying image {image_path}: {e}")
    
    def emotion_detection_loop(self):
        """
        Continuous emotion detection loop that runs every 2 seconds using camera
        """
        while self.running:
            try:
                # Detect emotion from camera
                new_emotion = self.detect_emotion_from_camera()
                
                if new_emotion and new_emotion != self.current_emotion:
                    print(f"🔮 Emotion changed to: {new_emotion}")
                    
                    # Get and display a random image for this emotion
                    image_path = self.get_random_image_for_emotion(new_emotion)
                    if image_path:
                        self.display_image(image_path)
                    else:
                        print(f"No images available for emotion: {new_emotion}")
                
                # Wait 5 seconds before next emotion check
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in emotion detection loop: {e}")
                time.sleep(2)
    
    def start_emotion_detection(self):
        """
        Start the emotion detection thread
        """
        if not self.emotion_thread or not self.emotion_thread.is_alive():
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop, daemon=True)
            self.emotion_thread.start()
            print("🔮 Emotion detection started (every 5 seconds)")
    
    def stop_emotion_detection(self):
        """
        Stop the emotion detection thread
        """
        self.running = False
        if self.emotion_thread:
            self.emotion_thread.join(timeout=1)
        print("🔮 Emotion detection stopped")
    
    def cleanup_camera(self):
        """
        Clean up camera resources
        """
        if self.camera:
            self.camera.release()
            self.camera = None
            print("🔮 Camera resources released")
    
    def init_camera(self):
        """
        Initialize camera for emotion detection with minimal resource usage
        """
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                # Set lower resolution for faster processing
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                # Reduce frame rate to save CPU
                self.camera.set(cv2.CAP_PROP_FPS, 5)
                print("🔮 Camera initialized for emotion detection")
            else:
                print("⚠️  Camera not available - emotion detection disabled")
                self.emotion_detection_enabled = False
                self.camera = None
        except Exception as e:
            print(f"⚠️  Camera initialization failed: {e}")
            self.emotion_detection_enabled = False
            self.camera = None
    
    def detect_emotion_from_camera(self):
        """
        Detect emotion from camera with minimal computational overhead
        Returns one of: angry, disgust, fear, happy, neutral, sad, surprised
        """
        if not self.emotion_detection_enabled or not self.camera:
            return self.current_emotion
        
        current_time = time.time()
        
        # Only detect emotion every few seconds to save CPU
        if current_time - self.last_emotion_detection < self.emotion_detection_interval:
            return self.current_emotion
        
        try:
            # Capture a single frame
            ret, frame = self.camera.read()
            if not ret:
                return self.current_emotion
            
            # Frame skipping: only process every 3rd frame to save CPU
            self.frame_skip_counter += 1
            if self.frame_skip_counter < self.frame_skip_interval:
                return self.current_emotion
            
            self.frame_skip_counter = 0  # Reset counter
            
            # Resize frame for faster processing (already at 160x120 from camera)
            # frame = cv2.resize(frame, (160, 120))  # No need to resize, already correct size
            
            # Detect emotion using DeepFace (minimal processing)
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv'  # Fastest backend
            )
            
            if result and len(result) > 0:
                emotion_data = result[0]
                dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
                emotion_scores = emotion_data.get('emotion', {})
                
                # Get confidence score for the dominant emotion
                confidence = emotion_scores.get(dominant_emotion, 0) / 100.0
                
                # Only update if confidence is above threshold
                if confidence >= self.emotion_confidence_threshold:
                    self.current_emotion = dominant_emotion
                    print(f"🔮 Emotion detected: {dominant_emotion} (confidence: {confidence:.2f})")
                else:
                    print(f"🔮 Low confidence emotion: {dominant_emotion} (confidence: {confidence:.2f})")
            
            self.last_emotion_detection = current_time
            
        except Exception as e:
            print(f"🔮 Emotion detection error: {e}")
            # Don't update emotion on error, keep current one
        
        return self.current_emotion
    
    def get_openai_response(self, user_text, emotion=None):
        """
        Get a response from OpenAI API using the magical mirror prompt
        """
        if not self.openai_client:
            return "I sense only silence. Speak your heart's desire, and I shall respond."

        try:
            print("🔮 The Mirror is consulting the ancient magic...")
            emotion_instruction = f"\nThe user currently appears {emotion}. Adjust your response tone accordingly." if emotion else ""
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.mirror_instructions + emotion_instruction},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=50,
                temperature=0.8,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"🔮 Error: {e}")
            return f"I sense your words: '{user_text}'. What deeper truth does your heart conceal?"
    
    def process_user_input(self, user_text, emotion=None):
        """
        Process user input and generate a mystical response using OpenAI
        """
        if not user_text.strip():
            return "I sense only silence. Speak your heart's desire, and I shall respond."
        
        # Use OpenAI for intelligent responses
        return self.get_openai_response(user_text, emotion)
    
    def handle_events(self):
        """
        Handle pygame events (for development - allows easy exit)
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n🔮 Window closed - Mirror fades away...")
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("\n🔮 ESC pressed - Mirror fades away...")
                    return False
        return True
    
    def run(self):
        """
        Main interactive loop for the Mirror of Erised
        """
        # Start emotion detection in background
        self.start_emotion_detection()
        
        # Initial greeting
        self.greet_user()
        
        try:
            while True:
                # Handle pygame events (for development)
                if not self.handle_events():
                    break
                
                # Record user speech with voice activity detection
                audio = self.record_with_voice_activity_detection()
                
                if audio is not None and len(audio) > 0:
                    # Transcribe the audio
                    user_text = self.transcribe_audio(audio)
                    
                    if user_text.strip():
                        print(f"🔮 You said: {user_text}")

                        emotion = self.detect_emotion_from_camera()
                        if emotion:
                            print(f"🔮 The Mirror perceives your emotion: {emotion}")
                        
                        # Process and respond
                        response = self.process_user_input(user_text, emotion)
                        self.speak(response)
                    else:
                        print("🔮 The Mirror heard only silence...")
                        self.speak("I sense only silence. Speak your heart's desire.")
                
                # Brief pause before listening again
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🔮 The Mirror fades away... Farewell.")
            self.speak("The Mirror of Erised bids you farewell. Until we meet again in your dreams.")
        finally:
            # Clean up
            self.stop_emotion_detection()
            self.cleanup_camera()
            if self.display_screen:
                pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Mirror of Erised Interactive Voice Interface")
    parser.add_argument("--whisper-model", default="tiny-q5_1", help="Whisper model name (default: tiny-q5_1)")
    parser.add_argument("--piper-voice", default="en_GB-alan-medium.onnx", help="Piper voice model path")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)")
    parser.add_argument("--silence-threshold", type=float, default=0.01, help="Silence detection threshold (default: 0.01)")
    parser.add_argument("--silence-duration", type=float, default=1.75, help="Silence duration before stopping (default: 1.75s)")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini for speed)")
    parser.add_argument("--images-dir", default="emotion_images", help="Directory containing emotion-based images (default: emotion_images)")
    parser.add_argument("--display-width", type=int, default=1024, help="Display width in pixels (default: 1024 for 10-inch LCD)")
    parser.add_argument("--display-height", type=int, default=600, help="Display height in pixels (default: 600 for 10-inch LCD)")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode (default: windowed for development)")
    parser.add_argument("--tts-speed", type=float, default=1.3, 
                    help="TTS speed multiplier (1.0 = normal, 1.3 = 30% faster, 1.5 = 50% faster)")
    
    args = parser.parse_args()
    
    # Create and run the Mirror of Erised
    mirror = MirrorOfErised(
        whisper_model=args.whisper_model,
        piper_voice=args.piper_voice,
        sample_rate=args.sample_rate,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        images_dir=args.images_dir,
        display_width=args.display_width,
        display_height=args.display_height,
        fullscreen=args.fullscreen,
        tts_speed=args.tts_speed
    )
    
    mirror.run()

if __name__ == "__main__":
    main()