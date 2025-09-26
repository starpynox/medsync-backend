import os
import base64
import io
from flask import Flask, request, jsonify
import pdfplumber
import cohere

app = Flask(__name__)

# Get API key from environment variable
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Set the COHERE_API_KEY environment variable!")

co = cohere.Client(COHERE_API_KEY)

@app.route("/", methods=["POST"])
def summarize_pdf():
    try:
        data = request.get_json()
        if not data or "pdf_base64" not in data:
            return jsonify({"error": "Missing 'pdf_base64' in request"}), 400

        pdf_bytes = base64.b64decode(data["pdf_base64"])
        pdf_file = io.BytesIO(pdf_bytes)

        # Extract text and tables
        extracted_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                extracted_text += f"\n--- Page {i} ---\n"
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
                tables = page.extract_tables()
                if tables:
                    for t_index, table in enumerate(tables, start=1):
                        extracted_text += f"\nTable {t_index}:\n"
                        for row in table:
                            extracted_text += ", ".join([str(cell) for cell in row]) + "\n"

        if not extracted_text.strip():
            return jsonify({"error": "No text found in PDF"}), 400

        # Call Cohere for summarization
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Summarize in simple terms:\n" + extracted_text}]
        )

        summary = response.choices[0].message["content"]
        return jsonify({"summary": summary})

    except Exception as e:
        # Return the error to frontend
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
