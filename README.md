# Fake News Detection using BERT and LSTM

## Overview
This project implements a fake news detection system using a hybrid deep learning model combining BERT and LSTM. The system classifies news articles as real or fake based on textual content. It is built with TensorFlow and Keras and deployed using a Flask API for real-time predictions.

## Features
- **BERT Tokenization:** Preprocesses text using BERT tokenizer.
- **Deep Learning Model:** Combines BERT embeddings with a bidirectional LSTM for classification.
- **Real-Time Predictions:** Flask-based API to classify user-provided news articles.
- **Efficient Training:** Optimized using TensorFlow with tuned hyperparameters.

## Dataset
- **Fake News:** Data from `Fake.csv`
- **Real News:** Data from `True.csv`
- Labels: `0` for Fake, `1` for Real

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the BERT model:
   ```sh
   from transformers import BertTokenizer, TFBertModel
   BertTokenizer.from_pretrained('bert-base-uncased')
   TFBertModel.from_pretrained('bert-base-uncased')
   ```

## Usage
### Training the Model
Run the training script to preprocess data and train the model:
```sh
python train.py
```

### Running the Flask API
```sh
python app.py
```
The API will be available at `http://127.0.0.1:5000/predict`.

### Making a Prediction
Send a POST request with JSON input:
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "Your news article text here"}'
```
Response:
```json
{
  "result": "Real",
  "confidence": 0.89
}
```

## Model Architecture
- **Input Layer:** Tokenized text input (BERT embeddings)
- **BERT Encoder:** Extracts contextual embeddings
- **LSTM Layer:** Captures sequential dependencies
- **Dropout Layer:** Prevents overfitting
- **Dense Layer:** Outputs binary classification

## License
This project is open-source and available under the MIT License.

