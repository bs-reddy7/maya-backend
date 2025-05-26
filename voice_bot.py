import os
import requests
import sounddevice as sd
import numpy as np
import threading
import time
import tempfile
from scipy.io.wavfile import write
from dotenv import load_dotenv
import whisper
import queue
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
ELEVENLABS_API_KEY = "sk_a37c5cfeb89d0e2ab57a34dd1fdd187cd996d09e04389a69"
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

# === Ultra-fast audio parameters ===
RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 15
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 0
SILENCE_DURATION_MS = 150
MIN_SPEECH_DURATION_MS = 250

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
    
    def speak_optimized(self, text, emotion="neutral"):
        """Optimized speech"""
        try:
            if not text or not text.strip():
                return False
            
            audio_data = self.synthesize_streaming(text, emotion)
            if not audio_data:
                print("⚠️ No audio generated.")
                return False
            
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            samples = np.array(audio_segment.get_array_of_samples())
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
            
            sd.play(samples, samplerate=audio_segment.frame_rate)
            sd.wait()
            
            return True
            
        except Exception as e:
            print(f"❌ ElevenLabs playback error: {e}")
            return False
    
    def get_usage_stats(self):
        """Get current usage statistics"""
        session_duration = datetime.now() - self.session_start
        return {
            "characters_used": self.character_count,
            "cache_hits": len(self.audio_cache),
            "session_duration": str(session_duration).split('.')[0],
            "estimated_cost": self.character_count * 0.00018
        }

# === Fast Speech Recognition ===
class FastASR:
    def __init__(self):
        print("🎤 Loading optimized Whisper for speed...")
        self.asr_model = whisper.load_model("base", device="cpu")
        self.vad = webrtcvad.Vad(1)
        self.audio_queue = queue.Queue()
        self.is_listening = True
        print("✅ Fast Whisper ASR ready")
    
    def stop_listening(self):
        """Stop listening to prevent feedback"""
        self.is_listening = False
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        print("🔇 Stopped listening")
    
    def start_listening(self):
        """Resume listening"""
        self.is_listening = True
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        print("🎤 Resumed listening")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback"""
        try:
            if self.is_listening:
                audio_np = np.frombuffer(indata, dtype=np.int16)
                self.audio_queue.put(audio_np.tobytes())
        except Exception:
            pass
    
    def detect_speech_fast(self, timeout_seconds=10):
        """Speech detection"""
        if not self.is_listening:
            return None
            
        frames = []
        silence_threshold = int(1500 / CHUNK_DURATION_MS)
        min_speech_frames = int(600 / CHUNK_DURATION_MS)
        timeout_frames = int(timeout_seconds * 1000 / CHUNK_DURATION_MS)
        
        silence_frames = 0
        speech_frames = 0
        frame_count = 0
        triggered = False

        while not self.audio_queue.empty():
            try: 
                self.audio_queue.get_nowait()
            except: 
                break

        print("🎤 Listening...")
        start_time = time.time()
        
        while frame_count < timeout_frames and self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.02)
                frame_count += 1
                
                audio_np = np.frombuffer(chunk, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
                peak_amplitude = np.max(np.abs(audio_np))
                
                is_speech = (energy > 120 and peak_amplitude > 400) or energy > 250
                
                if is_speech:
                    if not triggered:
                        print("🗣️ Speaking...")
                        triggered = True
                    frames.append(chunk)
                    speech_frames += 1
                    silence_frames = 0
                else:
                    if triggered:
                        silence_frames += 1
                        frames.append(chunk)
                        
                        if (silence_frames >= silence_threshold and 
                            speech_frames >= min_speech_frames):
                            print("✅ Got your message")
                            break
                            
            except queue.Empty:
                if triggered and silence_frames >= silence_threshold // 2:
                    if speech_frames >= min_speech_frames:
                        break
                continue

        if not frames or speech_frames < min_speech_frames:
            return None
            
        detection_time = time.time() - start_time
        print(f"🎤 Captured in {detection_time:.1f}s")
        return b''.join(frames)
    
    def transcribe_fast(self, audio_bytes):
        """Fast transcription"""
        if not audio_bytes:
            return ""
        try:
            start_time = time.time()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
                
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            write(path, RATE, audio_np)
            
            result = self.asr_model.transcribe(
                path, 
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
            os.unlink(path)
            
            transcription_time = time.time() - start_time
            
            text = result.get("text", "").strip()
            
            if text and len(text) > 3:
                text = text.replace("Thank you.", "").replace("Thanks for watching.", "").strip()
                
                words = text.split()
                if len(words) > 1 and len(text) > 5:
                    print(f"📝 You: '{text}' (⚡{transcription_time:.1f}s)")
                    return text
                else:
                    print(f"🔇 Filtered: '{text}'")
            
            return ""
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

# === SIMPLE APPROACH: Complete Generation Then Speech ===
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

# === Simple Maya Therapy Bot ===
class MayaTherapyBot:
    def __init__(self, api_key):
        print("🧘 Initializing Maya - Your Custom Therapy Companion...")
        
        if not api_key:
            print("❌ ELEVENLABS_API_KEY not found!")
            return
        
        # Always use Jessica voice
        jessica_voice_id = "cgSgspJ2msm6clMCkdW9"
        
        self.tts = ElevenLabsTTSClient(api_key, jessica_voice_id)
        self.asr = FastASR()
        self.llm = SimpleMayaLLM()  # ← Back to simple approach
        self.is_speaking = False
        self.stop_event = threading.Event()
        
        if not self.tts.test_connection():
            print("❌ Failed to connect to ElevenLabs API!")
            return
        
        print(f"✅ Maya ready with Jessica voice!")
        
    def speak_async(self, text, emotion="neutral"):
        def _speak():
            self.is_speaking = True
            self.asr.stop_listening()
            
            try:
                success = self.tts.speak_optimized(text, emotion)
                if not success:
                    print("❌ Speech synthesis failed")
            finally:
                time.sleep(0.5)
                self.is_speaking = False
                self.asr.start_listening()
                
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
        return thread
    
    def run_conversation(self):
        print("🧘 Maya - Your Personal Therapy Companion")
        print("🎯 Powered by your UPDATED gemma-companion:latest (1.7GB)")
        print("📦 Model ID: 5753d9fbc524")
        print("🎵 Speaking with Jessica's warm voice")
        print("💬 Complete generation → Clean speech (no interruptions)")
        
        print("🎤 Starting audio input...")
        with sd.RawInputStream(
            samplerate=RATE,
            blocksize=CHUNK_SIZE,
            dtype='int16',
            channels=CHANNELS,
            callback=self.asr.audio_callback
        ):
            # Welcome message
            welcome = "Hey there! I'm Maya. What's going on with you today?"
            print("🎵 Maya says:")
            self.speak_async(welcome, "neutral").join()
            
            print("\n🎙️ Ready to chat! Say 'goodbye' when you're done.")
            print("💬 Maya generates complete thoughts, then speaks clearly")
            print("🎯 Simple, reliable conversation with your updated model\n")

            silence_count = 0
            conversation_count = 0
            
            while not self.stop_event.is_set():
                try:
                    if conversation_count % 5 == 0 and conversation_count > 0:
                        stats = self.tts.get_usage_stats()
                        print(f"💰 Usage: {stats['characters_used']} chars, ~${stats['estimated_cost']:.4f}")
                    
                    if not self.is_speaking:
                        audio_data = self.asr.detect_speech_fast(timeout_seconds=8)
                        if not audio_data:
                            silence_count += 1
                            if silence_count >= 3:
                                encouragements = [
                                    "I'm here whenever you're ready to talk.",
                                    "Take your time. What's on your heart?",
                                    "I'm listening. What would you like to share?"
                                    ]
                                response = encouragements[conversation_count % len(encouragements)]
                                print(f"🎵 Maya says: '{response}'")
                                self.speak_async(response, "neutral").join()
                                silence_count = 0
                            continue
                    else:
                        time.sleep(0.2)
                        continue

                    silence_count = 0
                    conversation_count += 1

                    user_text = self.asr.transcribe_fast(audio_data)
                    if not user_text or len(user_text.strip()) < 2:
                        continue

                    # Exit
                    if any(w in user_text.lower().split() for w in ["goodbye", "bye", "exit", "quit"]):
                        farewell = "It was really good talking with you today. Take care of yourself, and I'm here whenever you need someone to listen. See you later!"
                        print(f"🎵 Maya says: '{farewell}'")
                        self.speak_async(farewell, "happy").join()
                        break

                    # Get complete response from updated Maya
                    print("🔄 Processing with your updated Maya model...")
                    response, emotion = self.llm.query_optimized(user_text)
                    
                    if response and response.strip():
                        # Speak the complete response
                        print(f"🎵 Maya says: '{response}'")
                        self.speak_async(response, emotion).join()
                        print("✅ Maya finished speaking, ready for your response...")
                    else:
                        fallback = "Tell me more about that."
                        print(f"🎵 Maya says: '{fallback}'")
                        self.speak_async(fallback, "calm").join()

                except KeyboardInterrupt:
                    print("\n👋 Chat ended")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

        # Final stats
        stats = self.tts.get_usage_stats()
        print(f"\n📊 Session Complete!")
        print(f"💰 Total characters used: {stats['characters_used']}")
        print(f"💳 Estimated cost: ${stats['estimated_cost']:.4f}")
        print(f"⏱️ Session duration: {stats['session_duration']}")
        print("🧘 Maya with updated model signing off! Great conversation!")

# === Main Function ===
def main():
    print("🧘 Maya - Your Personal Therapy Companion")
    print("🎯 Powered by YOUR Custom gemma-companion:latest")
    print("🎵 Jessica Voice + Sentence-by-Sentence Speech")
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
    
    print("\n🎵 Using Jessica voice for natural conversations")
    print("🎯 Sentence-by-sentence speech for perfect timing")
    print("💬 Maya speaks complete thoughts as they're ready")
    print("🚀 Starting conversation...")
    
    try:
        bot = MayaTherapyBot(api_key)
        bot.run_conversation()
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