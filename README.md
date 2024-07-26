# YouTube Video Popularity Prediction

## Predicting YouTube Video Popularity for Data Science, Data Analysis, Machine Learning and AI Categories

### Introduction
The YouTube Video Popularity Prediction project aims to predict the popularity of YouTube videos within the Data Science, Machine Learning, and AI categories. Users can make requests via an API key, and predictions are based on metrics such as likes, comments, publication date, and video title.

### Project Deployment
This project is deployed on Google Cloud, leveraging its infrastructure to provide reliable and scalable predictions.

### Live Application
You can access the live application at the following link: [YouTube Video Popularity Prediction App](https://videopopularityprediction-ofclhminyq-ue.a.run.app)

### Data Collection and Model Training
1. **Data Collection**: The dataset was collected using YouTube API keys to fetch video data.
2. **Data Storage**: The collected data was imported into a database for efficient storage and retrieval.
3. **Data Processing**: The raw data was processed and cleaned to be used for model training.
4. **Model Creation**: XGBoost model was created and trained using the processed data.
5. **Model Metrics**:
   - **Mean Squared Error (MSE)**: 0.2503614445288233
   - **Mean Absolute Error (MAE)**: 0.3286098056089632
   - **Root Mean Squared Error (RMSE)**: 0.5003613139810305
   - **R-squared**: 0.9440297650626439

### Web Server and UI
A web server was created using the Flask framework to allow users to interact with the model predictions through a user-friendly interface.

### Installation and Usage Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YanSav10/youtube_video_popularity_prediction
   cd youtube-video-popularity-prediction
2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv venv
   source venv/bin/activate
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the server**:
   ```bash
   python server/main.py
The server will be available at http://localhost:5000.
