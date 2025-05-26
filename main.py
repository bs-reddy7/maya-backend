from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from voice_bot import MayaBackendService, ELEVENLABS_API_KEY
import io

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize Maya's Backend Service ===
maya = MayaBackendService(api_key=ELEVENLABS_API_KEY)
latest_audio = None

# === Route: Process Audio Input ===
@app.post("/listen")
async def listen(audio: UploadFile = File(...)):
    global latest_audio
    try:
        audio_data = await audio.read()
        user_text, response_text, audio_response = maya.process_audio_input(audio_data)

        if isinstance(audio_response, str):  # If it's an error message
            return JSONResponse(status_code=500, content={"error": audio_response})

        if audio_response:
            latest_audio = audio_response
            return {
                "user_text": user_text,
                "response_text": response_text,
                "audio_url": "/audio"
            }

        return JSONResponse(status_code=400, content={"error": "Failed to process audio"})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Route: Serve Audio File ===
@app.get("/audio")
def get_audio():
    if latest_audio:
        return StreamingResponse(io.BytesIO(latest_audio), media_type="audio/mpeg")
    return JSONResponse(status_code=404, content={"error": "No audio available"})

# === Route: Text Input (optional)
@app.post("/text")
async def text_input(payload: dict):
    try:
        text_input = payload.get("text", "").strip()
        if not text_input:
            return JSONResponse(status_code=400, content={"error": "Text input is empty"})

        response_text, audio_response = maya.process_text_input(text_input)

        if isinstance(audio_response, str):
            return JSONResponse(status_code=500, content={"error": audio_response})

        if audio_response:
            global latest_audio
            latest_audio = audio_response
            return {
                "response_text": response_text,
                "audio_url": "/audio"
            }

        return JSONResponse(status_code=400, content={"error": "No response generated"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Route: Usage Stats ===
@app.get("/stats")
def stats():
    return maya.get_usage_stats()
