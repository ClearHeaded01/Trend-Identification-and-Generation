# from flask import Flask, request, jsonify
# import requests

# app = Flask(__name__)

# @app.route('/api/identify-trend', methods=['POST'])
# def identify_trend():
#     data = request.json
#     # Call your Trend Identification model API
#     response = requests.post('http://model-host/identify', json=data)
#     return jsonify(response.json())

# @app.route('/api/recommend-trend', methods=['POST'])
# def recommend_trend():
#     data = request.json
#     # Call your Trend Recommendation model API
#     response = requests.post('http://model-host/recommend', json=data)
#     return jsonify(response.json())

# @app.route('/api/generate-trend', methods=['POST'])
# def generate_trend():
#     data = request.json
#     # Call your Trend Generation model API
#     response = requests.post('http://model-host/generate', json=data)
#     return jsonify(response.json())

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import tensorflow_hub as hub
import pickle

from image_stylization import load_image

#from image_stylization import load_image

app = Flask(__name__)

# Load trend identification model
trend_identification_model = RandomForestRegressor(n_estimators=100, random_state=42)
trend_identification_model = pickle.load(open('top_trends.pkl','rb'))

# Load trend recommendation model
trend_recommendation_model = NearestNeighbors(n_neighbors = 10, algorithm='brute' ,metric = 'euclidean')
trend_recommendation_model = pickle.load(open('recommendation_modal.pkl' , 'rb'))

# Load trend generation model
trend_generation_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

@app.route('/')
def index():
    return render_template('index.html')
# @app.route('/trends')
# def recommend_ui():
#     return render_template('recommend.html')

@app.route('/trends', methods=['POST'])
def identify_trends():
    data = request.get_json()
    # Preprocess data
    df = pd.DataFrame(data)
    # Run trend identification model
    predicted_units_sold = trend_identification_model.predict(df)
    # Return top trends
    top_trends = pd.DataFrame({'title_orig_tokenized': df['title_orig_tokenized'], 'predicted_units_sold': predicted_units_sold})
    top_trends = top_trends.sort_values(by='predicted_units_sold', ascending=False).head(10)
    return jsonify(top_trends.to_dict(orient='records'))

@app.route('/recommendations', methods=['POST'])
def recommend_styles():
    top_trends = request.get_json()
    # Run trend recommendation model
    recommended_styles = trend_recommendation_model.predict(top_trends)
    return jsonify(recommended_styles)

@app.route('/generate', methods=['POST'])
def generate_trend_image():
    content_image_url = request.form['content_image_url']
    style_image_url = request.form['style_image_url']
    # Load images
    content_image = load_image(content_image_url , (244, 244))
    style_image = load_image(style_image_url,(244, 244) )
    # Run trend generation model
    stylized_image = trend_generation_model(content_image, style_image)
    # Save image to file
    image_path = 'generated_image.jpg'
    tf.keras.utils.save_img(image_path, stylized_image)
    return send_file(image_path, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
