from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from google import genai
import os

app = Flask(__name__)
CORS(app)

# Configuration
GOOGLE_API_KEY = os.getenv("API_KEY")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
COLLECTION_NAME = "ChatData"
PROMPT = os.getenv("PROMPT")

PROMPT = PROMPT.replace("*", "").replace("\n", "\n")

def connect_db():
    try:
        return psycopg2.connect(CONNECTION_STRING)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def search_documents(question, db):
    context = []
    docs = db.similarity_search(question, k=10)
    for doc in docs:
        context.append(doc.page_content)
    return context

def initialize_model():
    client = genai.Client(api_key=GOOGLE_API_KEY)
    chats = client.chats.create(model="gemini-2.0-flash")
    response = chats.send_message(PROMPT)
    return chats

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, 
                                              model="models/text-embedding-004")
    db = PGVector(embeddings, connection=CONNECTION_STRING, collection_name=COLLECTION_NAME, use_jsonb=True)
    
    model = initialize_model()
    context = search_documents(user_input, db)
    
    message = f'''
            Context: {context}
            Question: {user_input}
            '''
    
    response = model.send_message(message)
    return jsonify({"response": response.text.replace("*", "")})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Use PORT from Render, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)