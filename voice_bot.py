import os
import requests
import numpy as np
import threading
import time
import tempfile
from scipy.io.wavfile import write
from dotenv import load_dotenv
import whisper
import webrtcvad
from pydub import AudioSegment
import base64
import io
import hashlib
import json
from datetime import datetime, timedelta

# === Load environment variables ===
print(f"📁 Working directory: {os.getcwd()}")
print(f"📄 .env file exists: {os.path.exists('.env')}")

# Manual .env loading for debugging
ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"
LLM_API_URL = "http://localhost:11434/api/generate"

# Try multiple methods to load API key
try:
    # Method 1: Try dotenv
    load_dotenv()
    ELEVENLABS_API_KEY = os.getenv("sk_a37c5cfeb89d0e2ab57a34dd1fdd187cd996d09e04389a69")
    if ELEVENLABS_API_KEY:
        print(f"✅ Method 1 - dotenv: {ELEVENLABS_API_KEY[:8]}...{ELEVENLABS_API_KEY[-4:]}")
    else:
        print("❌ Method 1 - dotenv failed")
    
    # Method 2: Manual file reading
    if not ELEVENLABS_API_KEY and os.path.exists('.env'):
        print("🔍 Trying manual .env file reading...")
        with open('.env', 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📝 Raw .env content: {repr(content)}")
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('ELEVENLABS_API_KEY='):
                    ELEVENLABS_API_KEY = line.split('=', 1)[1].strip()
                    # Remove quotes if present
                    if ELEVENLABS_API_KEY.startswith('"') and ELEVENLABS_API_KEY.endswith('"'):
                        ELEVENLABS_API_KEY = ELEVENLABS_API_KEY[1:-1]
                    if ELEVENLABS_API_KEY.startswith("'") and ELEVENLABS_API_KEY.endswith("'"):
                        ELEVENLABS_API_KEY = ELEVENLABS_API_KEY[1:-1]
                    print(f"✅ Method 2 - manual: {ELEVENLABS_API_KEY[:8]}...{ELEVENLABS_API_KEY[-4:]}")
                    break
    
    # Method 3: Direct environment check
    if not ELEVENLABS_API_KEY:
        ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
        if ELEVENLABS_API_KEY:
            print(f"✅ Method 3 - direct env: {ELEVENLABS_API_KEY[:8]}...{ELEVENLABS_API_KEY[-4:]}")
    
except Exception as e:
    print(f"❌ Error loading API key: {e}")

# Final check
if not ELEVENLABS_API_KEY:
    print("❌ No API key found with any method")
else:
    print(f"🎯 Final API key: {ELEVENLABS_API_KEY[:8]}...{ELEVENLABS_API_KEY[-4:]}")

# === Audio parameters ===
RATE = 16000
CHANNELS = 1

# === Optimized ElevenLabs TTS Client ===
class ElevenLabsTTSClient:
    def __init__(self, api_key, voice_id="cgSgspJ2msm6clMCkdW9"):  # Jessica voice hardcoded
        self.api_key = api_key
        self.voice_id = voice_id  # Always Jessica
        self.base_url = "https://api.elevenlabs.io/v1"
        self.character_count = 0
        self.audio_cache = {}
        self.session_start = datetime.now()
        
    def test_connection(self):
        """Test ElevenLabs API connection"""
        try:
            headers = {"xi-api-key": self.api_key}
            response = requests.get(f"{self.base_url}/voices", headers=headers, timeout=5)
            
            if response.status_code == 200:
                voices = response.json()
                print(f"✅ ElevenLabs connected - {len(voices.get('voices', []))} voices available")
                return True
            else:
                print(f"❌ ElevenLabs API error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ElevenLabs connection failed: {e}")
            return False
    
    def optimize_text(self, text):
        """Clean up text"""
        if not text:
            return ""
        
        optimized = text.strip()
        optimized = ' '.join(optimized.split())
        optimized = optimized.replace('...', '.')
        
        return optimized
    
    def get_cache_key(self, text, voice_id):
        """Generate cache key for audio"""
        return hashlib.md5(f"{text}_{voice_id}".encode()).hexdigest()
    
    def synthesize_streaming(self, text, emotion="neutral"):
        """Streaming TTS for real-time feel"""
        try:
            optimized_text = self.optimize_text(text)
            if not optimized_text:
                return None
            
            # Always use Jessica voice
            voice_id = "cgSgspJ2msm6clMCkdW9"
            cache_key = self.get_cache_key(optimized_text, voice_id)
            
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]
            
            start_time = time.time()
            
            url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
            
            headers = {
                "Accept": "audio/mpeg",
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": optimized_text,
                "model_id": "eleven_flash_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": False
                },
                "output_format": "mp3_44100_128"
            }
            
            print(f"🎵 Jessica: '{optimized_text}' ({len(optimized_text)} chars)")
            
            response = requests.post(url, json=data, headers=headers, timeout=10, stream=True)
            
            if response.status_code == 200:
                audio_data = b""
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_data += chunk
                
                generation_time = time.time() - start_time
                print(f"⚡ Generated in {generation_time:.3f}s")
                
                self.audio_cache[cache_key] = audio_data
                self.character_count += len(optimized_text)
                
                return audio_data
                
            else:
                print(f"❌ ElevenLabs error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ ElevenLabs synthesis error: {e}")
            return None
    
    def get_audio_data(self, text, emotion="neutral"):
        """Get audio data without playing (for API responses)"""
        try:
            if not text or not text.strip():
                return None
            
            audio_data = self.synthesize_streaming(text, emotion)
            if not audio_data:
                print("⚠️ No audio generated.")
                return None
            
            return audio_data
            
        except Exception as e:
            print(f"❌ ElevenLabs audio generation error: {e}")
            return None
    
    def get_usage_stats(self):
        """Get current usage statistics"""
        session_duration = datetime.now() - self.session_start
        return {
            "characters_used": self.character_count,
            "cache_hits": len(self.audio_cache),
            "session_duration": str(session_duration).split('.')[0],
            "estimated_cost": self.character_count * 0.00018
        }

# === Speech Recognition (for processing uploaded audio) ===
class AudioProcessor:
    def __init__(self):
        print("🎤 Loading Whisper for transcription...")
        self.asr_model = whisper.load_model("base", device="cpu")
        print("✅ Whisper ASR ready")
    
    def transcribe_audio_file(self, audio_file_path):
        """Transcribe audio file uploaded from frontend"""
        try:
            start_time = time.time()
            
            result = self.asr_model.transcribe(
                audio_file_path, 
                language="en",
                fp16=False, 
                verbose=False,
                temperature=0.0,
                word_timestamps=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                beam_size=1,
                best_of=1,
                patience=1.0
            )
            
            transcription_time = time.time() - start_time
            
            text = result.get("text", "").strip()
            
            if text and len(text) > 3:
                text = text.replace("Thank you.", "").replace("Thanks for watching.", "").strip()
                
                words = text.split()
                if len(words) > 1 and len(text) > 5:
                    print(f"📝 Transcribed: '{text}' (⚡{transcription_time:.1f}s)")
                    return text
                else:
                    print(f"🔇 Filtered: '{text}'")
            
            return ""
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def transcribe_audio_bytes(self, audio_bytes):
        """Transcribe audio from bytes (for uploaded audio data)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            
            text = self.transcribe_audio_file(path)
            os.unlink(path)
            return text
            
        except Exception as e:
            print(f"❌ Audio bytes transcription error: {e}")
            return ""

# === Maya LLM (unchanged) ===
class SimpleMayaLLM:
    def __init__(self):
        self.model_name = "gemma-companion:latest"
        self.model_ready = self.test_connection()
        self.conversation_history = []
        self.last_responses = []
        
        # Minimal instant responses
        self.instant_responses = {
            "hello": ("Hi there! I'm Maya. How are you doing today?", "neutral"),
            "hi": ("Hey! What's been going on with you?", "neutral"),
            "hey": ("Hello! What's on your mind?", "neutral"),
        }
    
    def test_connection(self):
        """Test your updated custom model"""
        try:
            print(f"🎯 Testing your updated model: {self.model_name}")
            print("📦 Model ID: 5753d9fbc524 (1.7 GB)")
            
            test_payload = {
                "model": self.model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {
                    "num_ctx": 4096,    # Higher context for your updated model
                    "temperature": 0.8,
                    "num_predict": 10   # Short test
                }
            }
            
            response = requests.post(LLM_API_URL, json=test_payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                test_response = result.get("response", "").strip()
                print(f"🔥 {self.model_name} connected successfully!")
                print(f"📝 Test response: '{test_response}'")
                print("🎯 Using your updated Maya personality!")
                return True
            else:
                print(f"❌ {self.model_name} connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False
    
    def detect_emotion(self, text):
        """Simple emotion detection"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["sad", "depressed", "down", "terrible", "awful"]):
            return "sad"
        elif any(word in text_lower for word in ["anxious", "worried", "stressed", "nervous"]):
            return "anxious"
        elif any(word in text_lower for word in ["angry", "mad", "frustrated"]):
            return "angry"
        elif any(word in text_lower for word in ["happy", "great", "excited", "amazing"]):
            return "happy"
        else:
            return "neutral"
    
    def generate_complete_response(self, user_input):
        """Generate complete response before speaking"""
        
        if not self.model_ready:
            return "I'm having trouble connecting right now. Can you try again?"
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Build context
        recent_context = ""
        if len(self.conversation_history) > 1:
            recent_history = self.conversation_history[-6:]
            recent_context = "\n".join(recent_history) + "\n"
        
        # Simple prompt for your custom model
        prompt = f"""{recent_context}User: {user_input}
Maya:"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # ← COMPLETE GENERATION FIRST
                "options": {
                    "num_ctx": 4096,        # Higher context for updated model
                    "temperature": 0.9,     # Natural variety
                    "top_p": 0.95,
                    "num_predict": 100,     # Let Maya speak freely, no restrictions
                    "stop": ["\nUser:", "\nHuman:", "User:", "Human:"],
                    "repeat_penalty": 1.1
                }
            }
            
            print(f"🧠 Generating complete response with updated {self.model_name}...")
            start_time = time.time()
            
            response = requests.post(LLM_API_URL, json=payload, timeout=45)  # Longer timeout for complete generation
            
            if response.status_code == 200:
                result = response.json()
                full_response = result.get("response", "").strip()
                
                generation_time = time.time() - start_time
                
                if full_response:
                    print(f"✅ Generated in {generation_time:.2f}s: '{full_response}'")
                    
                    # Store conversation history
                    self.conversation_history.append(f"Maya: {full_response}")
                    self.last_responses.append(full_response)
                    
                    if len(self.last_responses) > 5:
                        self.last_responses.pop(0)
                    
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-8:]
                    
                    return full_response
                else:
                    print("⚠️ Empty response from model")
                    return "Tell me more about that."
                
            else:
                print(f"❌ Model error: {response.status_code}")
                return "I'm having some technical difficulties. Can you repeat that?"
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "I had a small hiccup. What were you saying?"
    
    def query_optimized(self, user_input):
        """Get complete response from Maya"""
        start_time = time.time()
        
        emotion = self.detect_emotion(user_input)
        
        # Check for instant responses
        user_lower = user_input.lower().strip()
        word_count = len(user_input.split())
        
        if word_count <= 2:
            for keyword, (response, response_emotion) in self.instant_responses.items():
                if user_lower == keyword or user_lower.startswith(keyword + " "):
                    response_time = time.time() - start_time
                    print(f"⚡ Instant response ({response_time:.3f}s)")
                    return response, response_emotion
        
        # Generate complete response
        print(f"🔄 Using Maya for: '{user_input}' ({word_count} words)")
        response = self.generate_complete_response(user_input)
        
        response_time = time.time() - start_time
        print(f"🧠 Maya response ({response_time:.2f}s)")
        
        return response, emotion

# === Backend API Service (for web integration) ===
class MayaBackendService:
    def __init__(self, api_key):
        print("🧘 Initializing Maya Backend Service...")
        
        if not api_key:
            print("❌ ELEVENLABS_API_KEY not found!")
            return
        
        # Always use Jessica voice
        jessica_voice_id = "cgSgspJ2msm6clMCkdW9"
        
        self.tts = ElevenLabsTTSClient(api_key, jessica_voice_id)
        self.audio_processor = AudioProcessor()
        self.llm = SimpleMayaLLM()
        
        if not self.tts.test_connection():
            print("❌ Failed to connect to ElevenLabs API!")
            return
        
        print(f"✅ Maya Backend Service ready with Jessica voice!")
    
    def process_audio_input(self, audio_data):
        """Process audio input from frontend and return text + audio response"""
        try:
            # 1. Transcribe the audio
            print("🎤 Processing audio input...")
            user_text = self.audio_processor.transcribe_audio_bytes(audio_data)
            
            if not user_text or len(user_text.strip()) < 2:
                return None, None, "No speech detected"
            
            # 2. Generate Maya's response
            print(f"🔄 Processing: '{user_text}'")
            response_text, emotion = self.llm.query_optimized(user_text)
            
            if not response_text:
                return user_text, None, "No response generated"
            
            # 3. Generate audio for the response
            print(f"🎵 Generating audio for: '{response_text}'")
            audio_response = self.tts.get_audio_data(response_text, emotion)
            
            if not audio_response:
                return user_text, response_text, "Audio generation failed"
            
            return user_text, response_text, audio_response
            
        except Exception as e:
            print(f"❌ Error processing audio input: {e}")
            return None, None, f"Error: {e}"
    
    def process_text_input(self, text_input):
        """Process text input and return text + audio response"""
        try:
            if not text_input or len(text_input.strip()) < 2:
                return None, "Text input too short"
            
            # Generate Maya's response
            print(f"🔄 Processing text: '{text_input}'")
            response_text, emotion = self.llm.query_optimized(text_input)
            
            if not response_text:
                return None, "No response generated"
            
            # Generate audio for the response
            print(f"🎵 Generating audio for: '{response_text}'")
            audio_response = self.tts.get_audio_data(response_text, emotion)
            
            if not audio_response:
                return response_text, "Audio generation failed"
            
            return response_text, audio_response
            
        except Exception as e:
            print(f"❌ Error processing text input: {e}")
            return None, f"Error: {e}"
    
    def get_usage_stats(self):
        """Get current usage statistics"""
        return self.tts.get_usage_stats()

# === Flask API Example (optional) ===
def create_flask_app():
    """Example Flask app for web integration"""
    try:
        from flask import Flask, request, jsonify, send_file
        from flask_cors import CORS
        import io
        
        app = Flask(__name__)
        CORS(app)
        
        # Initialize Maya service
        api_key = ELEVENLABS_API_KEY
        if not api_key:
            print("❌ No API key for Flask app!")
            return None
        
        maya_service = MayaBackendService(api_key)
        
        @app.route('/api/chat/audio', methods=['POST'])
        def chat_audio():
            """Handle audio input from frontend"""
            try:
                if 'audio' not in request.files:
                    return jsonify({'error': 'No audio file provided'}), 400
                
                audio_file = request.files['audio']
                audio_data = audio_file.read()
                
                user_text, response_text, audio_response = maya_service.process_audio_input(audio_data)
                
                if isinstance(audio_response, str):  # Error message
                    return jsonify({'error': audio_response}), 500
                
                if audio_response:
                    # Return JSON with text and audio URL
                    return jsonify({
                        'user_text': user_text,
                        'response_text': response_text,
                        'audio_url': '/api/audio/latest'  # Endpoint to get audio
                    })
                else:
                    return jsonify({'error': 'Failed to generate response'}), 500
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/chat/text', methods=['POST'])
        def chat_text():
            """Handle text input from frontend"""
            try:
                data = request.json
                text_input = data.get('text', '').strip()
                
                if not text_input:
                    return jsonify({'error': 'No text provided'}), 400
                
                response_text, audio_response = maya_service.process_text_input(text_input)
                
                if isinstance(audio_response, str):  # Error message
                    return jsonify({'error': audio_response}), 500
                
                if audio_response:
                    return jsonify({
                        'response_text': response_text,
                        'audio_url': '/api/audio/latest'
                    })
                else:
                    return jsonify({'error': 'Failed to generate response'}), 500
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Store latest audio for retrieval
        latest_audio = None
        
        @app.route('/api/audio/latest', methods=['GET'])
        def get_latest_audio():
            """Get the latest generated audio"""
            if latest_audio:
                return send_file(
                    io.BytesIO(latest_audio),
                    mimetype='audio/mpeg',
                    as_attachment=False
                )
            else:
                return jsonify({'error': 'No audio available'}), 404
        
        @app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get usage statistics"""
            stats = maya_service.get_usage_stats()
            return jsonify(stats)
        
        return app
        
    except ImportError:
        print("❌ Flask not installed. Install with: pip install flask flask-cors")
        return None

# === Main Function (for testing) ===
def main():
    print("🧘 Maya Backend Service")
    print("🎯 No mic recording - Frontend handles audio input")
    print("🎵 ElevenLabs TTS + Whisper ASR + Custom Maya LLM")
    print("=" * 55)
    
    # Check for API key
    api_key = ELEVENLABS_API_KEY
    if not api_key:
        print("❌ No API key found")
        api_key = input("🔑 Enter your ElevenLabs API key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    else:
        print("✅ Using API key from environment")
    
    try:
        # Initialize backend service
        maya_service = MayaBackendService(api_key)
        
        # Example usage
        print("\n🧪 Testing text input processing...")
        response_text, audio_data = maya_service.process_text_input("Hello Maya, how are you?")
        
        if response_text:
            print(f"✅ Response: {response_text}")
            print(f"🎵 Audio generated: {len(audio_data) if audio_data else 0} bytes")
        
        # Flask app example
        print("\n🌐 Starting Flask API server...")
        app = create_flask_app()
        if app:
            print("✅ Flask app ready!")
            print("📡 Endpoints:")
            print("   POST /api/chat/audio - Send audio file")
            print("   POST /api/chat/text - Send text message")
            print("   GET /api/audio/latest - Get latest audio response")
            print("   GET /api/stats - Get usage statistics")
            print("\n🚀 Run with: app.run(host='0.0.0.0', port=5000, debug=True)")
        else:
            print("❌ Flask app initialization failed")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check ElevenLabs API key")
        print("   2. Start Ollama: ollama serve")
        print("   3. Load model: ollama run gemma-companion:latest")

if __name__ == "__main__":
    main()