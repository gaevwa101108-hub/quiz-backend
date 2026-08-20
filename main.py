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
You are an exhaustive, line-by-line educational quiz generator.
Your objective is 100% complete coverage of all provided document pages.

STRICT GRANULAR GENERATION RULES:
1. QUANTITY MANDATE: Generate AT LEAST 100 HIGH-QUALITY QUESTIONS PER PAGE scanned.
   - 1 Page scanned = Minimum 100 questions.
   - 2 Pages scanned = Minimum 200 questions.
   - 3 Pages scanned = Minimum 300 questions.
   - 4 Pages scanned = Minimum 400 questions.

2. ATOMIC EXTRACTION RULES:
   - PARAGRAPHS & TEXT: Generate AT LEAST 1 multiple-choice question for EVERY SINGLE SENTENCE in the textbook text. Do not summarize or skip any sentence.
   - TABLES & CHARTS: Generate AT LEAST 1 unique question for EVERY SINGLE CELL, entry, element, formula, or root name inside every table.
   - DIAGRAMS & FIGURES: Generate AT LEAST 1 question for every labeled item, caption, arrow, and key concept in diagrams or callout boxes.
   - DEFINITIONS & EXAMPLES: Convert every bold term, numerical example, or sample problem into a standalone question.

AGE-APPROPRIATE STYLE:
- Early childhood (K1-G3): Short, clear, simple text for read-alouds.
- Grades 4 to College (G4-C4): Complete, precise academic questions.

VISUALS:
- Include inline SVG diagram strings in the "svg" field ONLY when visual assistance is required. Otherwise set "svg": "".

Return ONLY a valid JSON array of objects using this exact schema:
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
    
    for file in files:
        file_bytes = await file.read()
        if file.content_type == "application/pdf":
            import io
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""
            contents_list.append(f"\n--- PDF PAGE ---\n{pdf_text}")
        else:
            mime = file.content_type if file.content_type.startswith("image/") else "image/jpeg"
            contents_list.append({"mime_type": mime, "data": file_bytes})

    # Output ceiling raised to 32,768 tokens for massive multi-page question generation
    generation_config = {
        "response_mime_type": "application/json",
        "max_output_tokens": 32768
    }
    model = genai.GenerativeModel('gemini-3.6-flash', generation_config=generation_config)

    try:
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
