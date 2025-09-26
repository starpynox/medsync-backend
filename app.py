from flask import Flask, request, jsonify
import pdfplumber
import base64
import io
import cohere
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load Cohere API key from environment variable
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable not set")

co = cohere.Client(COHERE_API_KEY)

@app.route("/", methods=["POST"])
def summarize_pdf():
    try:
        data = request.get_json()
        pdf_base64 = data.get("pdf_base64")

        if not pdf_base64:
            return jsonify({"error": "No PDF provided"}), 400

        # Decode the base64 PDF
        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_file = io.BytesIO(pdf_bytes)

        # Extract text from PDF
        output = ""
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                output += f"\n--- Page {i} ---\n"
                text = page.extract_text()
                if text:
                    output += text + "\n"

        # Send to Cohere for summary
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Summarize and explain in simple terms:\n" + output}]
        )

        summary = response.output_text

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
