from flask import Flask, jsonify, request
from flask_cors import CORS

from sample import chat_completion

import os

app = Flask(__name__)
CORS(app)  # This will allow CORS for all routes

api = chat_completion()
# Define a route that accepts POST requests and returns data
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json
    print("user_input: ", user_input['key'])
    input_text = user_input['key']
    response = api.get_model_response(input_text)
    return jsonify({
        "received": response
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)