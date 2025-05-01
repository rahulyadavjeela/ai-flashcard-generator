from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from werkzeug.utils import secure_filename
import PyPDF2
import docx

app = Flask(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "mistral:instruct"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB


def extract_text_from_file(filepath):
    try:
        if filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filepath.endswith('.docx'):
            return " ".join([para.text for para in docx.Document(filepath).paragraphs if para.text])
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"File Error: {str(e)}")
        return ""


def generate_flashcards(text, num_cards=3):
    prompt = f"""Generate exactly {num_cards} high-quality flashcards from this text. Follow these rules:
1. Create distinct cards about different concepts
2. Use this exact JSON format:
[{{"question": "What is...", "answer": "..."}}]
3. Cover the most important concepts first

Text: {text[:800]}"""  # Strict 800 character limit

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "format": "json",
                "options": {
                    "temperature": 0.4,
                    "num_ctx": 1024,
                    "num_predict": 300,
                    "top_k": 40
                },
                "stream": False
            },
            timeout=90  # Increased timeout
        )
        response.raise_for_status()

        # Process response
        data = response.json()
        content = data.get("response", "")

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()

        cards = json.loads(content)
        if not isinstance(cards, list):
            cards = [cards]  # Ensure we always return a list

        print(f"Generated {len(cards)} flashcards")
        return cards[:num_cards]

    except Exception as e:
        print(f"Generation failed: {str(e)}")
        # Return sample flashcards if API fails
        sample_cards = [
            {"question": "What is superintelligence?", "answer": "Hypothetical AI surpassing human intelligence"},
            {"question": "Name one benefit of AI", "answer": "Increased efficiency through automation"},
            {"question": "What is an AI ethical concern?", "answer": "Issues around bias and privacy"}
        ]
        return sample_cards[:num_cards]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    text = request.form.get('text', '').strip()

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                text = extract_text_from_file(filepath) or text
                os.remove(filepath)
            except Exception as e:
                return jsonify({"error": str(e)})

    if not text:
        return jsonify({"error": "No text provided"})

    try:
        # Get the number with default 3
        num_cards = int(request.form.get('num_cards', 3))

        # Apply constraints (1-5)
        num_cards = max(1, min(num_cards, 5))
        cards = generate_flashcards(text, num_cards)
        return jsonify({"cards": cards})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)