# --- Required imports ---
import os
import sys





from dotenv import load_dotenv


# --- Expose FastAPI app at module level for Uvicorn ---
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.append(BACKEND_DIR)

from backend.model import CattleDiseaseModel
from backend.utils import save_upload_file

from backend.gemini import GeminiAI
from fastapi import Body
from backend.milk_yield_api import router as milk_yield_router

MODEL_PATH = os.path.join('Diseases_Detection_Model', 'Model', 'cattle_cnn.pth')
TEST_FOLDER = os.path.join('Diseases_Detection_Model', 'Disease Data', 'test')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = CattleDiseaseModel(MODEL_PATH)
gemini_ai = GeminiAI()

# Mount milk yield prediction API
app.include_router(milk_yield_router)

from fastapi import Request

@app.post("/chat/gemini/")
async def chat_gemini(request: Request):
    """Chat with Google Gemini LLM. Uses a cattle-specialized system prompt. Accepts JSON: {prompt: str, history: [str]}"""
    body = await request.json()
    prompt = body.get("prompt", "")
    history = body.get("history")
    result = gemini_ai.chat(prompt, history=history)
    return result

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    save_path = os.path.join(TEST_FOLDER, file.filename)
    save_upload_file(file, save_path)
    disease = model.predict(save_path)
    return {"filename": file.filename, "prediction": disease}

@app.get("/predict_all/")
def predict_all():
    results = []
    for fname in os.listdir(TEST_FOLDER):
        fpath = os.path.join(TEST_FOLDER, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            try:
                pred = model.predict(fpath)
                results.append({"filename": fname, "prediction": pred})
            except Exception as e:
                results.append({"filename": fname, "error": str(e)})
    return {"results": results}
