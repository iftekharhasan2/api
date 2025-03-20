from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from dotenv import load_dotenv

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load and preprocess intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Prepare patterns for vectorization
all_patterns = []
pattern_to_intent = {}

for intent in intents['intents']:
    for pattern in intent['patterns']:
        if pattern.strip():  # Only add non-empty patterns
            all_patterns.append(pattern)
            pattern_to_intent[pattern] = intent

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(all_patterns)

def translate_to_bangla(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a translation engine that mimics Google Translate exactly."},
                {"role": "user", "content": "Translate the following text into Bengali exactly like Google Translate: " + text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def find_best_match(user_input, threshold=0.6):
    # Vectorize user input
    user_vector = vectorizer.transform([user_input])
    
    # Calculate similarities
    similarities = cosine_similarity(user_vector, pattern_vectors)
    
    # Find best match
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx]
    
    if best_match_score >= threshold:
        matched_pattern = all_patterns[best_match_idx]
        return pattern_to_intent[matched_pattern], best_match_score
    return None, best_match_score

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint that handles user messages"""
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({
                'error': 'No message provided',
                'status': 'error'
            }), 400

        # Try to find a match in intents
        intent_match, confidence = find_best_match(user_message)
        
        if intent_match and confidence > 0.7:
            # Use predefined response from intents
            response = np.random.choice(intent_match['responses'])
            # Translate the intent response to Bangla
            translated_response = translate_to_bangla(response)
            return jsonify({
                'response': translated_response,
                'original_response': response,
                'source': 'intents',
                'confidence': float(confidence),
                'status': 'success'
            })
        else:
            # Fall back to ChatGPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot in bangla."},
                    {"role": "user", "content": user_message}
                ]
            )
            return jsonify({
                'response': response.choices[0].message.content,
                'source': 'chatgpt',
                'status': 'success'
            })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/api/intents', methods=['GET'])
def get_intents():
    """Get all available intents"""
    try:
        return jsonify({
            'intents': intents['intents'],
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch intents',
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True) 