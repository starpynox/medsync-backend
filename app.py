from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import cohere
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize Cohere client
co = cohere.Client("YOUR_COHERE_API_KEY")  # Replace with your Cohere API key

@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    try:
        # Expecting JSON with base64-encoded PDF
        data = request.json
        pdf_base64 = data.get("pdf_base64", None)
        if not pdf_base64:
            return jsonify({"error": "No PDF provided"}), 400

        # Decode PDF
        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_file = BytesIO(pdf_bytes)

        # Extract text from PDF
        output_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    output_text += f"\n--- Page {i} ---\n{text}\n"

        if not output_text.strip():
            return jsonify({"error": "No text found in PDF"}), 400

        # Send text to Cohere for summarization
        response = co.chat(
            model="command-a-03-2025",
            message="Summarize and explain in simple terms: " + output_text
        )

        return jsonify({"summary": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
