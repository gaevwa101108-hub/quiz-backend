import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
import pypdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SYNC_FILE = "cloud_quiz_backups.json"

def load_sync_data():
    if os.path.exists(SYNC_FILE):
        try:
            with open(SYNC_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_sync_data(data):
    try:
        with open(SYNC_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Error writing sync data:", e)

PROMPT = """
You are an expert educational flashcard creator.
Analyze the provided document/image thoroughly.

STRICT INSTRUCTIONS:
1. Extract and convert AT LEAST 90% of all information and sentences into multiple-choice quiz questions.
2. AGE-APPROPRIATE QUESTION LENGTH:
   - For early childhood and primary grades (K1, K2, G1, G2, G3): Keep question text VERY SHORT, punchy, engaging, and simple to read or listen to out loud (e.g., "Which shape is red?", "How many apples?", "What comes next?").
   - For older grades (G4-G12, C1-C4): Provide complete, accurate academic questions.
3. PRIORITIZE:
   - Highlighted text (yellow markers, annotations)
   - Handwritten marginal notes or underlines
   - Section headers, titles, dates, names, chemical formulas, and key definitions
   - Cause-and-effect explanations
4. VISUALS, DIAGRAMS & FORMULAS:
   - Create clean, valid, standalone inline SVG code strings in the "svg" field whenever visual representation helps:
     * K1 to G3: Simple counting icons, shapes, colored objects, or basic visual diagrams.
     * Geometry/Math/Physics: Shapes, angles, force vectors, coordinate grids.
     * Chemistry: Chemical bonds, molecular structures.
   - If NO visual is needed, set "svg": "".

Return ONLY a valid JSON array containing objects with this exact schema:
[
  {
    "id": "Q1",
    "topic": "General",
    "q": "Very short question text here?",
    "svg": "",
    "correct": "Correct Answer",
    "options": ["Correct Answer", "Wrong Option 1", "Wrong Option 2", "Wrong Option 3"]
  }
]
Output raw JSON only. Do not wrap in markdown code blocks.
"""

@app.post("/api/scan")
async def scan_document(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")

    contents = await file.read()
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        if file.content_type.startswith("image/"):
            image_part = {"mime_type": file.content_type, "data": contents}
            response = model.generate_content([PROMPT, image_part])
        elif file.content_type == "application/pdf":
            import io
            pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""
            
            combined_prompt = f"{PROMPT}\n\nPDF TEXT CONTENT:\n{pdf_text}"
            response = model.generate_content(combined_prompt)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        raw_text = response.text.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]

        questions = json.loads(raw_text.strip())
        return JSONResponse(content={"questions": questions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Failed: {str(e)}")

@app.post("/api/sync/upload")
async def upload_sync(payload: dict = Body(...)):
    profile = payload.get("profile")
    quizzes = payload.get("quizzes")
    if not profile or quizzes is None:
        raise HTTPException(status_code=400, detail="Invalid payload.")
    
    db = load_sync_data()
    db[profile] = quizzes
    save_sync_data(db)
    return {"status": "success", "message": f"Cloud backup updated for {profile}."}

@app.get("/api/sync/download/{profile}")
async def download_sync(profile: str):
    db = load_sync_data()
    quizzes = db.get(profile, {})
    return {"quizzes": quizzes}

@app.get("/")
def read_root():
    return HTMLResponse(content="<h1>AI Quiz Generator Server is Running!</h1>")