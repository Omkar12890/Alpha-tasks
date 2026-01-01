// DOM Elements
const sourceText = document.getElementById('sourceText');
const sourceLang = document.getElementById('sourceLang');
const targetLang = document.getElementById('targetLang');
const translateBtn = document.getElementById('translateBtn');
const clearBtn = document.getElementById('clearBtn');
const swapBtn = document.getElementById('swapBtn');
const charCount = document.getElementById('charCount');
const resultContainer = document.getElementById('resultContainer');
const errorContainer = document.getElementById('errorContainer');
const originalText = document.getElementById('originalText');
const translatedText = document.getElementById('translatedText');
const errorMessage = document.getElementById('errorMessage');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');

// Language data
let languages = {};

// Load languages on page load
async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        languages = await response.json();
        populateLanguageSelects();
    } catch (error) {
        console.error('Failed to load languages:', error);
        showError('Failed to load available languages');
    }
}

// Populate language dropdowns
function populateLanguageSelects() {
    sourceLang.innerHTML = '<option value="auto">Auto-detect</option>';
    targetLang.innerHTML = '';

    for (const [code, name] of Object.entries(languages)) {
        const optionSource = document.createElement('option');
        optionSource.value = code;
        optionSource.textContent = name;
        sourceLang.appendChild(optionSource);

        const optionTarget = document.createElement('option');
        optionTarget.value = code;
        optionTarget.textContent = name;
        if (code === 'es') optionTarget.selected = true;
        targetLang.appendChild(optionTarget);
    }
}

// Character counter
sourceText.addEventListener('input', () => {
    charCount.textContent = sourceText.value.length;
    if (sourceText.value.length > 5000) {
        sourceText.value = sourceText.value.substring(0, 5000);
        charCount.textContent = '5000';
    }
});

// Translate button click
translateBtn.addEventListener('click', translate);

// Clear button click
clearBtn.addEventListener('click', () => {
    sourceText.value = '';
    charCount.textContent = '0';
    resultContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');
    sourceLang.value = 'auto';
    targetLang.value = 'es';
});

// Swap languages button
swapBtn.addEventListener('click', () => {
    const source = sourceLang.value;
    const target = targetLang.value;
    
    if (source !== 'auto') {
        sourceLang.value = target;
        targetLang.value = source;
        
        // If there's translated text, swap and translate again
        if (translatedText.textContent) {
            sourceText.value = translatedText.textContent;
            setTimeout(() => translate(), 100);
        }
    }
});

// Copy buttons
document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const targetId = btn.dataset.target;
        const text = document.getElementById(targetId).textContent;
        copyToClipboard(text, btn);
    });
});

// Speak buttons
document.querySelectorAll('.speak-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const targetId = btn.dataset.target;
        const text = document.getElementById(targetId).textContent;
        speakText(text, targetLang.value);
    });
});

// Enter key to translate
sourceText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        translate();
    }
});

// Main translate function
async function translate() {
    const text = sourceText.value.trim();
    const source = sourceLang.value;
    const target = targetLang.value;

    // Validation
    if (!text) {
        showError('Please enter text to translate');
        return;
    }

    if (source === target && source !== 'auto') {
        showError('Source and target languages must be different');
        return;
    }

    // Show loading state
    setLoading(true);
    errorContainer.classList.add('hidden');

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                source: source,
                target: target
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Translation failed');
        }

        // Display results
        originalText.textContent = data.original;
        translatedText.textContent = data.translated;
        resultContainer.classList.remove('hidden');
        errorContainer.classList.add('hidden');

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred during translation');
    } finally {
        setLoading(false);
    }
}

// Helper function to show error
function showError(message) {
    errorMessage.textContent = message;
    errorContainer.classList.remove('hidden');
    resultContainer.classList.add('hidden');
}

// Helper function to set loading state
function setLoading(loading) {
    translateBtn.disabled = loading;
    if (loading) {
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
    } else {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
}

// Copy to clipboard function
function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = '✓ Copied!';
        setTimeout(() => {
            button.textContent = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        showError('Failed to copy text');
    });
}

// Text-to-speech function
function speakText(text, language) {
    // Cancel any ongoing speech
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }

    // Map language codes to speech synthesis language codes
    const languageMap = {
        'en': 'en-US',
        'es': 'es-ES',
        'fr': 'fr-FR',
        'de': 'de-DE',
        'it': 'it-IT',
        'pt': 'pt-BR',
        'ja': 'ja-JP',
        'zh-CN': 'zh-CN',
        'ko': 'ko-KR',
        'ru': 'ru-RU'
    };

    const speechLang = languageMap[language] || 'en-US';

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = speechLang;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Handle speech synthesis errors
    utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        showError('Text-to-speech is not supported in your browser');
    };

    window.speechSynthesis.speak(utterance);
}

// Initialize - load languages and focus on input
window.addEventListener('load', () => {
    loadLanguages();
    sourceText.focus();
    console.log('Language Translation Tool loaded successfully');
});
