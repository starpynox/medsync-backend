from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import cohere
import os

# Initialize Cohere client
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

app = Flask(__name__)
CORS(app)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Extract text, tables, images
    data = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_data = {
                "page_number": i,
                "text": page.extract_text(),
                "tables": page.extract_tables(),
                "images": page.images
            }
            data.append(page_data)

    # Build combined string
    output = ""
    for page in data:
        output += f"\n--- Page {page['page_number']} ---\n"
        if page["text"]:
            output += f"\nText:\n{page['text']}\n"
        if page["tables"]:
            for t_index, table in enumerate(page["tables"], start=1):
                output += f"\nTable {t_index}:\n"
                for row in table:
                    output += ", ".join(str(cell) for cell in row) + "\n"
        if page["images"]:
            output += f"\nImages ({len(page['images'])} found)\n"

    # Send to Cohere
    try:
        response = co.chat(
            model="command-a-03-2025",
            message="Summarize and explain in simple terms: " + output
        )
        summary = response.text
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)
