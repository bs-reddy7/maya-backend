from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from voice_bot import MayaTherapyBot

# === Create bot instance ===
bot = MayaTherapyBot(api_key=os.getenv("ELEVENLABS_API_KEY", "sk_a37c5cfeb89d0e2ab57a34dd1fdd187cd996d09e04389a69"))

app = FastAPI()

# === Enable CORS for frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to frontend URL on Vercel later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Ensure audio folder exists ===
AUDIO_FOLDER = "audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

@app.post("/listen")
async def listen(audio: UploadFile = File(...)):
    try:
        # Save uploaded voice temporarily
        raw_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
        with open(raw_path, "wb") as f:
            f.write(await audio.read())

        # Transcribe user message
        user_input = bot.asr.transcribe_fast(open(raw_path, "rb").read())
        if not user_input:
            return JSONResponse({"error": "Could not transcribe input."}, status_code=400)

        # Get LLM response + emotion
        response_text, emotion = bot.llm.query_optimized(user_input)

        # Synthesize voice
        audio_data = bot.tts.synthesize_streaming(response_text, emotion)
        if not audio_data:
            return JSONResponse({"error": "TTS generation failed."}, status_code=500)

        output_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.mp3")
        with open(output_path, "wb") as f:
            f.write(audio_data)

        return {
            "text": response_text,
            "audio_url": f"/audio/{os.path.basename(output_path)}"
        }

    except Exception as e:
        return JSONResponse({"error": f"Something went wrong: {str(e)}"}, status_code=500)

@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(AUDIO_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse({"error": "Audio file not found"}, status_code=404)
