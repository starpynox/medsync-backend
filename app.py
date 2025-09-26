import os
import pdfplumber
import cohere
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Cohere API key from environment variable
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY environment variable is not set!")

co = cohere.Client(COHERE_API_KEY)

app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Replace "*" with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "✅ PDF Summarizer API is running"}

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    # Save uploaded PDF temporarily
    contents = await file.read()
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Extract text, tables, and images
    output = ""
    with pdfplumber.open(temp_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            output += f"\n--- Page {i} ---\n"
            text = page.extract_text()
            if text:
                output += f"\nText:\n{text}\n"
            tables = page.extract_tables()
            if tables:
                for t_index, table in enumerate(tables, start=1):
                    output += f"\nTable {t_index}:\n"
                    for row in table:
                        output += ", ".join(str(cell) for cell in row) + "\n"
            if page.images:
                output += f"\nImages ({len(page.images)} found)\n"

    os.remove(temp_path)

    # Send to Cohere for summarization
    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {"role": "user", "content": "Summarize and explain in simple terms: " + output}
        ]
    )

    summary_text = response['choices'][0]['message']['content']
    return {"summary": summary_text}
