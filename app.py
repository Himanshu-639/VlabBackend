from flask import Flask, request, jsonify
import re
import nltk
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# --- Independent Text Processing Functions ---

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

def lemmatize(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

def preprocess_text(text):
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


# --- Similarity Calculation Function ---

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(sim, 4)


# --- Flask Route to Handle Processing ---

@app.route('/preprocess', methods=['POST'])
def preprocess_endpoint():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    steps1 = {
        "lower": lowercase(text1),
        "punct": remove_punctuation(lowercase(text1)),
        "stop": remove_stopwords(remove_punctuation(lowercase(text1))),
        "lemma": lemmatize(remove_stopwords(remove_punctuation(lowercase(text1))))
    }

    steps2 = {
        "lower": lowercase(text2),
        "punct": remove_punctuation(lowercase(text2)),
        "stop": remove_stopwords(remove_punctuation(lowercase(text2))),
        "lemma": lemmatize(remove_stopwords(remove_punctuation(lowercase(text2))))
    }

    return jsonify({
        "step1": steps1,
        "step2": steps2
    })

@app.route('/similarity', methods=['POST'])
def similarity_endpoint():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    similarity = compute_similarity(text1, text2)
    return jsonify({ "similarity": similarity })

@app.route('/similarity-matrix', methods=['POST'])
def similarity_matrix():
    data = request.json
    texts = data.get('texts', [])

    if not texts or len(texts) < 2:
        return jsonify({"error": "Provide at least two texts"}), 400

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    matrix = cosine_similarity(tfidf_matrix)

    # Convert matrix (numpy array) to nested Python list for JSON serialization
    similarity_matrix = matrix.tolist()

    return jsonify({"similarity_matrix": similarity_matrix})

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(debug=True)
