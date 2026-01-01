const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const translate = require('translate-google');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public'));

// Supported languages
const LANGUAGES = {
  'en': 'English',
  'es': 'Spanish',
  'fr': 'French',
  'de': 'German',
  'it': 'Italian',
  'pt': 'Portuguese',
  'ja': 'Japanese',
  'zh-CN': 'Chinese (Simplified)',
  'ko': 'Korean',
  'ru': 'Russian'
};

// Routes
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/public/index.html');
});

app.get('/api/languages', (req, res) => {
  res.json(LANGUAGES);
});

app.post('/api/translate', async (req, res) => {
  try {
    const { text, source, target } = req.body;

    if (!text || !text.trim()) {
      return res.status(400).json({ error: 'Please enter text to translate' });
    }

    if (source === target && source !== 'auto') {
      return res.status(400).json({ error: 'Source and target languages must be different' });
    }

    // Perform translation
    const result = await translate(text, {
      from: source === 'auto' ? 'auto' : source,
      to: target
    });

    res.json({
      original: text,
      translated: result,
      source: source,
      target: target,
      success: true
    });

  } catch (error) {
    console.error('Translation error:', error);
    res.status(500).json({
      error: 'Translation failed: ' + error.message,
      success: false
    });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🌍 Language Translation Tool running at http://localhost:${PORT}`);
  console.log('Press Ctrl+C to stop the server');
});
