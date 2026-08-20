import os
import json
from typing import List
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
You are an expert pedagogical AI educator.
Your task is to analyze the provided page image(s) or document file(s) as a single, continuous educational lesson.

STUDY & COMPREHENSION MANDATE:
1. First, study the entire material holistically. If multiple pages are provided, connect continuous thoughts, rules, or sentences that cross page boundaries.
2. Extract and transform AT LEAST 95% of all information—including body text, sidebars, diagrams, callout boxes, and itemized entries inside tables or charts—into comprehensive multiple-choice questions.
3. Do not gloss over structured data (tables/lists). Treat every data row or conceptual pairing as an essential fact to be tested.
4. Target Output Depth: For dense academic pages, generate 30 to 50 thorough, non-redundant questions that test both direct factual retrieval and conceptual application.

AGE-APPROPRIATE ADAPTATION:
- Kindergarten to Grade 3 (K1-G3): Keep question text VERY SHORT, punchy, and clear for audio read-alouds.
- Grades 4 to College (G4-C4): Provide complete, academically rigorous questions.

VISUALS & DIAGRAMS:
- Generate clean, standalone inline SVG code string in the "svg" field ONLY when visual assistance (geometry, grids, molecular structures, counting items) enhances comprehension. Otherwise, set "svg": "".

Return ONLY a valid JSON array using this exact schema:
[
  {
    "id": "Q1",
    "topic": "General",
    "q": "Question text here?",
    "svg": "",
    "correct": "Correct Answer",
    "options": ["Correct Answer", "Wrong Option 1", "Wrong Option 2", "Wrong Option 3"]
  }
]
"""

@app.post("/api/scan")
async def scan_documents(files: List[UploadFile] = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")

    contents_list = []
    
    # Process all uploaded files into Gemini parts
    for file in files:
        file_bytes = await file.read()
        if file.content_type == "application/pdf":
            import io
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""
            contents_list.append(f"\n--- PDF DOCUMENT PAGE ---\n{pdf_text}")
        else:
            # Handle image types (.jpg, .png, etc.)
            mime = file.content_type if file.content_type.startswith("image/") else "image/jpeg"
            contents_list.append({"mime_type": mime, "data": file_bytes})

    generation_config = {
        "response_mime_type": "application/json"
    }
    model = genai.GenerativeModel('gemini-3.6-flash', generation_config=generation_config)

    try:
        # Build prompt payload with all image/text parts
        prompt_parts = [PROMPT] + contents_list
        response = model.generate_content(prompt_parts)

        raw_text = response.text.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]

        questions = json.loads(raw_text.strip())
        return JSONResponse(content={"questions": questions})

    except Exception as e:
        print("Backend Error:", str(e))
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
