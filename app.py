from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

API_KEY = "symptomPred123_API"

MODEL_FILEPATH = "model.pkl"

with open(MODEL_FILEPATH, "rb") as f:
    gbm = pickle.load(f)
    state_dict = gbm.__getstate__()
    classes_array = state_dict["classes_"]
    features_dict = {
        feature: i for i, feature in enumerate(state_dict["feature_names_in_"])
    }

@app.route("/", methods=["GET"])
def default():
    return jsonify({"message": "Welcome to the MediAI API!"})

@app.route("/predict", methods=["POST"])
def predict():
    if not request.json or "symptoms" not in request.json:
        return jsonify({"message": "Invalid request, symptoms field is required"})

    # Check if the API key is provided and validate it
    if "api_key" not in request.headers or request.headers["api_key"] != API_KEY:
        return jsonify({"message": "Unauthorized"}), 401

    symptoms = request.json["symptoms"]

    if not symptoms:
        return jsonify({"output": []})

    coded_features = [
        features_dict[keyword] for keyword in symptoms if keyword in features_dict
    ]

    sample_x = np.zeros(len(features_dict), dtype=np.float32)
    sample_x[coded_features] = 1

    probs = gbm.predict_proba(sample_x.reshape(1, -1))[0]

    output = list(zip(classes_array, probs.astype(float)))
    output.sort(key=lambda x: x[1], reverse=True)

    return jsonify({"output": output[:3]})

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({"classes": classes_array.tolist(), "features": list(features_dict)})

# @app.route("/recommend", methods=["POST"])
# def recommend():
#     if not request.json or "symptoms" not in request.json:
#         return jsonify({"message": "Invalid request, symptoms field is required"})

#     symptoms = request.json["symptoms"]

#     # Perform recommendation logic using the loaded models
#     # Example: recommendation_result = recommendation_logic(symptoms, median_model, similarity_model)

#     # Return recommendation result
#     return jsonify({"recommendation": recommendation_result})

# RECOMMENDATION_MEDIAN_FILEPATH = "median.pkl"
# RECOMMENDATION_SIMILARITY_FILEPATH = "similarity.pkl"

# # Load recommendation system models
# with open(RECOMMENDATION_MEDIAN_FILEPATH, "rb") as f:
#     median_model = pickle.load(f)

# with open(RECOMMENDATION_SIMILARITY_FILEPATH, "rb") as f:
#     similarity_model = pickle.load(f)

                                                                                                                                                                              
if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import numpy as np
# from flask_pymongo import PyMongo

# app = Flask(__name__)
# CORS(app)

# API_KEY = "symptomPred123_API"

# MODEL_FILEPATH = "model.pkl"

# with open(MODEL_FILEPATH, "rb") as f:
#     gbm = pickle.load(f)
#     state_dict = gbm.__getstate__()
#     classes_array = state_dict["classes_"]
#     features_dict = {
#         feature: i for i, feature in enumerate(state_dict["feature_names_in_"])
#     }

# app.config["MONGO_URI"] = "mongodb://localhost:27017/Mydatabase"
# mongo = PyMongo(app)

# @app.route('/api/register', methods=['POST'])
# def register():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     # Ensure email and password are not empty
#     if not email or not password:
#         return jsonify({'error': 'Email and password are required'}), 400

#     # Example: Check if user already exists in MongoDB
#     existing_user = mongo.db.users.find_one({"email": email})
#     if existing_user:
#         return jsonify({'error': 'User already exists'}), 400

#     # Save user data to MongoDB
#     mongo.db.users.insert_one({"email": email, "password": password})

#     return jsonify({'message': 'User registered successfully'}), 200

# @app.route('/api/login', methods=['POST', 'OPTIONS'])
# def login():
#     if request.method == 'OPTIONS':
#         # Handle OPTIONS requests by returning the allowed methods and headers
#         response = app.make_default_options_response()
#         response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
#         response.headers['Access-Control-Allow-Headers'] = 'Content-Type, api_key'
#         return response

#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     # Ensure email and password are not empty
#     if not email or not password:
#         return jsonify({'error': 'Email and password are required'}), 400

#     # Your login logic goes here...

#     return jsonify({'message': 'Login successful'}), 200

# # Add your existing routes for data, predict, and metadata here...

# @app.route("/", methods=["GET"])
# def default():
#     return jsonify({"message": "Welcome to the MediAI API!"})

# if __name__ == "__main__":
#     app.run(debug=True)
