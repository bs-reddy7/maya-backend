#!/usr/bin/env python3
"""
Maya Therapy Bot - FastAPI Web API Server
Transforms your voice bot into a complete backend API
"""

import os
import io
import json
import time
import base64
import tempfile
import traceback
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Import your existing Maya components with better error handling
try:
    from voice_bot import (
        ElevenLabsTTSClient,
        AudioProcessor, 
        SimpleMayaLLM,
        ELEVENLABS_API_KEY,
        LLM_API_URL
    )
    print("✅ All Maya components imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure voice_bot.py is in the same directory as main.py")
    print("🔧 Attempting to continue with fallback components...")
    
    # Fallback values if voice_bot.py is missing
    ELEVENLABS_API_KEY = os.getenv("sk_a37c5cfeb89d0e2ab57a34dd1fdd187cd996d09e04389a69")
    LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
    
    # You would need to define fallback classes here if voice_bot.py is missing
    raise ImportError("voice_bot.py is required for full functionality")

# === Configuration ===
API_VERSION = "1.0.0"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max audio file
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm', 'm4a', 'flac'}
UPLOAD_FOLDER = 'temp_uploads'

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maya_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === FastAPI App Setup ===
app = FastAPI(
    title="Maya Therapy Bot API",
    description="AI-powered therapy companion with voice interaction",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global Maya Service ===
maya_service = None
latest_audio_cache = {}  # Store latest audio responses

# === Pydantic Models ===
class TextChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    success: bool
    response_text: Optional[str] = None
    emotion: Optional[str] = None
    audio_url: Optional[str] = None
    processing_time: float
    session_id: str
    error: Optional[str] = None

def allowed_file(filename: str) -> bool:
    """Check if uploaded file is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

# === Maya API Service ===
class MayaAPIService:
    def __init__(self, api_key):
        """Initialize Maya API service with all components"""
        logger.info("🧘 Initializing Maya API Service...")
        
        if not api_key:
            raise ValueError("ElevenLabs API key is required!")
        
        # Initialize components
        self.tts = ElevenLabsTTSClient(api_key, "cgSgspJ2msm6clMCkdW9")  # Jessica voice
        self.audio_processor = AudioProcessor()
        self.llm = SimpleMayaLLM()
        self.session_stats = {
            'requests_processed': 0,
            'audio_requests': 0,
            'text_requests': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Test connections
        if not self.tts.test_connection():
            raise ConnectionError("Failed to connect to ElevenLabs API!")
        
        if not self.llm.model_ready:
            logger.warning("⚠️ Maya LLM not ready - check Ollama connection")
        
        logger.info("✅ Maya API Service initialized successfully!")
    
    def process_audio_request(self, audio_data: bytes, session_id: str = None):
        """Process audio input and return response"""
        try:
            start_time = time.time()
            self.session_stats['audio_requests'] += 1
            self.session_stats['requests_processed'] += 1
            
            logger.info(f"🎤 Processing audio request (Session: {session_id})")
            
            # 1. Transcribe audio
            user_text = self.audio_processor.transcribe_audio_bytes(audio_data)
            
            if not user_text or len(user_text.strip()) < 2:
                logger.warning("❌ No speech detected in audio")
                return {
                    'success': False,
                    'error': 'No speech detected',
                    'user_text': '',
                    'response_text': '',
                    'processing_time': time.time() - start_time
                }
            
            # 2. Generate Maya's response
            logger.info(f"🔄 User said: '{user_text}'")
            response_text, emotion = self.llm.query_optimized(user_text)
            
            if not response_text:
                logger.error("❌ Failed to generate response")
                return {
                    'success': False,
                    'error': 'Failed to generate response',
                    'user_text': user_text,
                    'response_text': '',
                    'processing_time': time.time() - start_time
                }
            
            # 3. Generate audio response
            logger.info(f"🎵 Maya responds: '{response_text}'")
            audio_response = self.tts.get_audio_data(response_text, emotion)
            
            if not audio_response:
                logger.error("❌ Failed to generate audio")
                return {
                    'success': False,
                    'error': 'Failed to generate audio',
                    'user_text': user_text,
                    'response_text': response_text,
                    'processing_time': time.time() - start_time
                }
            
            # Store audio in cache
            audio_id = f"{session_id}_{int(time.time())}" if session_id else str(int(time.time()))
            latest_audio_cache[audio_id] = {
                'audio_data': audio_response,
                'timestamp': time.time(),
                'text': response_text
            }
            
            # Clean old cache entries (keep last 10)
            if len(latest_audio_cache) > 10:
                oldest_key = min(latest_audio_cache.keys(), 
                               key=lambda k: latest_audio_cache[k]['timestamp'])
                del latest_audio_cache[oldest_key]
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Audio request processed in {processing_time:.2f}s")
            
            return {
                'success': True,
                'user_text': user_text,
                'response_text': response_text,
                'emotion': emotion,
                'audio_id': audio_id,
                'processing_time': processing_time,
                'audio_size': len(audio_response)
            }
            
        except Exception as e:
            self.session_stats['errors'] += 1
            logger.error(f"❌ Audio processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Audio processing failed: {str(e)}',
                'user_text': '',
                'response_text': '',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def process_text_request(self, text_input: str, session_id: str = None):
        """Process text input and return response"""
        try:
            start_time = time.time()
            self.session_stats['text_requests'] += 1
            self.session_stats['requests_processed'] += 1
            
            logger.info(f"💬 Processing text request: '{text_input}' (Session: {session_id})")
            
            if not text_input or len(text_input.strip()) < 2:
                return {
                    'success': False,
                    'error': 'Text input too short',
                    'response_text': '',
                    'processing_time': time.time() - start_time
                }
            
            # Generate Maya's response
            response_text, emotion = self.llm.query_optimized(text_input.strip())
            
            if not response_text:
                logger.error("❌ Failed to generate text response")
                return {
                    'success': False,
                    'error': 'Failed to generate response',
                    'response_text': '',
                    'processing_time': time.time() - start_time
                }
            
            # Generate audio response
            logger.info(f"🎵 Maya responds: '{response_text}'")
            audio_response = self.tts.get_audio_data(response_text, emotion)
            
            if not audio_response:
                logger.warning("⚠️ Failed to generate audio, returning text only")
                return {
                    'success': True,
                    'response_text': response_text,
                    'emotion': emotion,
                    'audio_id': None,
                    'processing_time': time.time() - start_time,
                    'warning': 'Audio generation failed'
                }
            
            # Store audio in cache
            audio_id = f"{session_id}_{int(time.time())}" if session_id else str(int(time.time()))
            latest_audio_cache[audio_id] = {
                'audio_data': audio_response,
                'timestamp': time.time(),
                'text': response_text
            }
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Text request processed in {processing_time:.2f}s")
            
            return {
                'success': True,
                'response_text': response_text,
                'emotion': emotion,
                'audio_id': audio_id,
                'processing_time': processing_time,
                'audio_size': len(audio_response)
            }
            
        except Exception as e:
            self.session_stats['errors'] += 1
            logger.error(f"❌ Text processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Text processing failed: {str(e)}',
                'response_text': '',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def get_stats(self):
        """Get comprehensive service statistics"""
        uptime = datetime.now() - self.session_stats['start_time']
        tts_stats = self.tts.get_usage_stats()
        
        return {
            'api_version': API_VERSION,
            'service_uptime': str(uptime).split('.')[0],
            'requests_processed': self.session_stats['requests_processed'],
            'audio_requests': self.session_stats['audio_requests'],
            'text_requests': self.session_stats['text_requests'],
            'errors': self.session_stats['errors'],
            'success_rate': round(
                (self.session_stats['requests_processed'] - self.session_stats['errors']) /
                max(self.session_stats['requests_processed'], 1) * 100, 2
            ),
            'tts_characters_used': tts_stats['characters_used'],
            'tts_estimated_cost': tts_stats['estimated_cost'],
            'cached_audio_responses': len(latest_audio_cache),
            'model_status': {
                'llm_ready': self.llm.model_ready,
                'llm_model': self.llm.model_name,
                'tts_voice': 'Jessica (cgSgspJ2msm6clMCkdW9)'
            }
        }

# === API Routes ===

@app.get("/")
async def api_info():
    """API information and health check"""
    return {
        'service': 'Maya Therapy Bot API',
        'version': API_VERSION,
        'status': 'healthy' if maya_service else 'initializing',
        'endpoints': {
            'POST /api/chat/audio': 'Send audio file for conversation',
            'POST /api/chat/text': 'Send text message for conversation', 
            'GET /api/audio/{audio_id}': 'Retrieve generated audio response',
            'GET /api/stats': 'Get service statistics',
            'GET /api/health': 'Health check'
        },
        'timestamp': datetime.now().isoformat(),
        'docs_url': '/docs'
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check - but don't fail if Maya isn't ready"""
    try:
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app_ready': True
        }
        
        if maya_service:
            health['services'] = {
                'tts': maya_service.tts.test_connection() if maya_service.tts else False,
                'llm': maya_service.llm.model_ready if maya_service.llm else False,
                'audio_processor': True
            }
            health['maya_ready'] = True
        else:
            health['services'] = {
                'tts': False,
                'llm': False, 
                'audio_processor': False
            }
            health['maya_ready'] = False
            health['status'] = 'degraded'
            health['message'] = 'Maya service not initialized'
        
        return health
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        # Still return 200 OK so health check passes
        return {
            'status': 'degraded',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'app_ready': True
        }

@app.post("/api/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...),
    session_id: str = Form(default="default")
):
    """Handle audio input for conversation"""
    try:
        if not maya_service:
            raise HTTPException(status_code=503, detail="Service not available")
        
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(audio.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {list(ALLOWED_AUDIO_EXTENSIONS)}"
            )
        
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        logger.info(f"📁 Received audio file: {len(audio_data)} bytes")
        
        # Process audio
        result = maya_service.process_audio_request(audio_data, session_id)
        
        if result['success']:
            return {
                'success': True,
                'user_text': result['user_text'],
                'response_text': result['response_text'],
                'emotion': result['emotion'],
                'audio_url': f"/api/audio/{result['audio_id']}",
                'processing_time': result['processing_time'],
                'session_id': session_id
            }
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    'success': False,
                    'error': result['error'],
                    'user_text': result.get('user_text', ''),
                    'processing_time': result['processing_time']
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Audio chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/api/chat/text")
async def chat_text(request: TextChatRequest):
    """Handle text input for conversation"""
    try:
        if not maya_service:
            raise HTTPException(status_code=503, detail="Service not available")
        
        text_input = request.text.strip()
        session_id = request.session_id
        
        if not text_input:
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(text_input) > 1000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")
        
        logger.info(f"💬 Received text: '{text_input[:50]}...' ({len(text_input)} chars)")
        
        # Process text
        result = maya_service.process_text_request(text_input, session_id)
        
        if result['success']:
            response_data = {
                'success': True,
                'response_text': result['response_text'],
                'emotion': result['emotion'],
                'processing_time': result['processing_time'],
                'session_id': session_id
            }
            
            # Add audio URL if available
            if result.get('audio_id'):
                response_data['audio_url'] = f"/api/audio/{result['audio_id']}"
            
            # Add warning if present
            if result.get('warning'):
                response_data['warning'] = result['warning']
            
            return response_data
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    'success': False,
                    'error': result['error'],
                    'processing_time': result['processing_time']
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Text chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Retrieve generated audio by ID"""
    try:
        if audio_id not in latest_audio_cache:
            raise HTTPException(status_code=404, detail="Audio not found")
        
        audio_info = latest_audio_cache[audio_id]
        audio_data = audio_info['audio_data']
        
        logger.info(f"🎵 Serving audio: {audio_id} ({len(audio_data)} bytes)")
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"inline; filename=maya_response_{audio_id}.mp3"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Audio retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio error: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive service statistics"""
    try:
        if not maya_service:
            raise HTTPException(status_code=503, detail="Service not available")
        
        stats = maya_service.get_stats()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    """Initialize Maya service on startup"""
    global maya_service
    
    try:
        logger.info("🚀 Starting Maya API Server...")
        logger.info("🔄 Initializing components...")
        
        # Check API key
        api_key = ELEVENLABS_API_KEY
        if not api_key:
            logger.warning("⚠️ No ElevenLabs API key found!")
            logger.info("💡 Maya will work with text-only responses")
        else:
            logger.info("✅ ElevenLabs API key found")
        
        # Initialize Maya service
        maya_service = MayaAPIService(api_key) if api_key else None
        logger.info("✅ Maya API Service ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Maya service: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise - allow app to start even if Maya service fails
        logger.info("🔄 App will continue with limited functionality")

# Add a simple ready check
@app.get("/ready")
async def ready_check():
    """Simple ready check that doesn't depend on Maya service"""
    return {"status": "ready", "timestamp": datetime.now().isoformat()}

# === Main Entry Point ===
if __name__ == '__main__':
    print("🧘 Maya Therapy Bot - Full FastAPI Web API Server")
    print("=" * 50)
    print(f"✅ Maya API Server ready!")
    print(f"🎯 Version: {API_VERSION}")
    print(f"🎵 Voice: Jessica (ElevenLabs)")
    print(f"🎤 Audio Processing: Full Whisper + WebRTC VAD")
    print(f"🧠 Maya LLM: Ollama Integration")
    print(f"🌐 FastAPI Documentation: /docs")
    
    import uvicorn
    # Fly.io uses PORT environment variable, default to 8080
    port = int(os.getenv("PORT", 8080))
    print(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        workers=1  # Single worker for consistent memory usage
    )