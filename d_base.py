from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
import pickle
import numpy as np

app = Flask(_name_)
CORS(app)

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/Mydatabase"
mongo = PyMongo(app)

# Load the machine learning model and necessary data
MODEL_FILEPATH = "model.pkl"

with open(MODEL_FILEPATH, "rb") as f:
    gbm = pickle.load(f)
    state_dict = gbm._getstate_()
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

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Ensure email and password are not empty
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    # Check if user already exists in MongoDB
    existing_user = mongo.db.users.find_one({"email": email})
    if existing_user:
        return jsonify({'error': 'User already exists'}), 400

    # Save user data to MongoDB
    mongo.db.users.insert_one({"email": email, "password": password})

    return jsonify({'message': 'User data stored in MongoDB'}), 200

@app.route('/api/data', methods=['GET'])
def get_data():
    data = list(mongo.db.userData.find({}, {'_id': 0}))
    return jsonify(data)

if _name_ == '_main_':
    app.run(debug=True)