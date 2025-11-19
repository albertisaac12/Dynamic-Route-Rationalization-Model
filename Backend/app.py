from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path to import inference.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from inference import predict_route
except ImportError as e:
    print(f"Error importing inference module: {e}")
    print("Make sure you are running this script from the correct directory or that inference.py exists in the parent directory.")
    sys.exit(1)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Call the inference function
        result = predict_route(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
