from flask import Flask, render_template, request, jsonify
from googletrans import Translator
import threading
import logging

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize translator
translator = Translator()

# Supported languages
LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'zh-cn': 'Chinese (Simplified)',
    'ko': 'Korean',
    'ru': 'Russian'
}

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data.get('text', '').strip()
        source_lang = data.get('source', 'auto')
        target_lang = data.get('target', 'en')
        
        if not text:
            return jsonify({'error': 'Please enter text to translate'}), 400
        
        # Perform translation
        translation = translator.translate(text, src_language=source_lang, dest_language=target_lang)
        
        result = {
            'original': text,
            'translated': translation['text'],
            'source': source_lang,
            'target': target_lang,
            'success': True
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': f'Translation failed: {str(e)}', 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
