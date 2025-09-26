import os
import base64
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
import cohere

app = Flask(__name__)
CORS(app)

co = cohere.Client(os.environ.get("COHERE_API_KEY"))

@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    try:
        data = request.json
        pdf_base64 = data.get("pdf_base64")
        if not pdf_base64:
            return jsonify({"error": "No PDF provided"}), 400

        pdf_bytes = base64.b64decode(pdf_base64)
        
        # Save temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_bytes)

        pdf_data = []
        with pdfplumber.open("temp.pdf") as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_data = {
                    "page_number": i,
                    "text": page.extract_text(),
                    "tables": page.extract_tables(),
                    "images": page.images
                }
                pdf_data.append(page_data)

        # Convert to string for cohere
        content = ""
        for page in pdf_data:
            content += f"\n--- Page {page['page_number']} ---\n"
            if page["text"]:
                content += f"\nText:\n{page['text']}\n"
            if page["tables"]:
                for t_index, table in enumerate(page["tables"], start=1):
                    content += f"\nTable {t_index}:\n"
                    for row in table:
                        content += ", ".join(str(cell) for cell in row) + "\n"
            if page["images"]:
                content += f"\nImages ({len(page['images'])} found)\n"

        # Generate summary
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Summarize and explain simply:\n" + content}]
        )

        summary = response.output_text if hasattr(response, 'output_text') else response.text
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
