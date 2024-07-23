import numpy as np
import pandas as pd
from datetime import datetime
from gensim.models import Word2Vec
import nltk
import pickle
import xgboost as xgb
import requests
from flask import Flask, request, jsonify, render_template
from google.cloud import secretmanager

nltk.download('punkt')

# Load the trained XGBoost model from pickle file
model_path = 'video_popularity_prediction.pickle'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

booster = model.get_booster()
booster.save_model('video_popularity_prediction_new.model')
bst = xgb.Booster()
bst.load_model('video_popularity_prediction_new.model')

def access_secret_version(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

project = "hale-yew-429904-f0"
secret = "api_key"

api = access_secret_version(project, secret)

# Function to fetch video details from YouTube Data API
def get_video_details(api_key, video_url):
    video_id = video_url.split('v=')[-1]
    url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'items' in data and len(data['items']) > 0:
        video_data = data['items'][0]
        title = video_data['snippet']['title']
        like_count = int(video_data['statistics'].get('likeCount', 0))
        comment_count = int(video_data['statistics'].get('commentCount', 0))
        upload_date = video_data['snippet']['publishedAt'][:10]
        return title, like_count, comment_count, upload_date
    else:
        raise ValueError("Video not found or API quota exceeded")

# Function to preprocess user input
def preprocess_input(video_title, like_count, comment_count, upload_date):
    df = pd.DataFrame({
        'video_title': [video_title],
        'like_count': [like_count],
        'comment_count': [comment_count],
        'upload_date': [upload_date]
    })

    df['upload_date'] = pd.to_datetime(df['upload_date'])
    df['year'] = df['upload_date'].dt.year
    df['month'] = df['upload_date'].dt.month
    df['day'] = df['upload_date'].dt.day
    df['days_since_upload'] = (datetime.now() - df['upload_date']).dt.days
    df['video_title'] = df['video_title'].apply(nltk.word_tokenize)

    model_vec = Word2Vec(sentences=df['video_title'], vector_size=100, window=5, min_count=1, workers=4)

    def get_average_word2vec(tokens, model, vector_size):
        vector = np.zeros(vector_size)
        count = 0
        for token in tokens:
            if token in model.wv:
                vector += model.wv[token]
                count += 1
        if count > 0:
            vector /= count
        return vector

    vector_size = 100
    embeddings = df['video_title'].apply(lambda x: get_average_word2vec(x, model_vec, vector_size))
    embedding_df = pd.DataFrame(embeddings.tolist(), index=df.index)
    df = pd.concat([df[['like_count', 'comment_count', 'days_since_upload', 'year', 'month', 'day']], embedding_df], axis=1)

    if 'video_title' in df.columns:
        df.drop(columns=['video_title'], inplace=True)
    if 'upload_date' in df.columns:
        df.drop(columns=['upload_date'], inplace=True)

    return df

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    video_url = request.form['video_url']
    try:
        title, like_count, comment_count, upload_date = get_video_details(api, video_url)
        df_input = preprocess_input(title, like_count, comment_count, upload_date)
        data = xgb.DMatrix(df_input)
        prediction = bst.predict(data)
        predicted_views = float(np.round(np.exp(prediction)[0] / 10, 1))
        formatted_views = "{:,.1f}".format(predicted_views).replace(',', ' ')
        return jsonify({'predicted_views': formatted_views})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
