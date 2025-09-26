from flask import Flask, request, jsonify
import pdfplumber
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    summaries = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                prompt = f"Summarise and explain in simple terms (Page {i}):\n{text[:5000]}"
                response = model.generate_content(prompt)
                summaries.append(f"Page {i} Summary:\n{response.text}\n")
    
    # Combine summaries into one final summary
    final_summary_input = "\n".join(summaries)
    overall = model.generate_content("Give me one final summary:\n" + final_summary_input[:5000])
    
    return jsonify({"summary": overall.text})

if __name__ == "__main__":
    app.run(debug=True)