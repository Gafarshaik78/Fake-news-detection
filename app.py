import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Step 1: Load and preprocess datasets
fake_df = pd.read_csv('/content/Fake.csv')
true_df = pd.read_csv('/content/True.csv')

fake_df['label'] = 0  # Fake news
true_df['label'] = 1  # Real news

data = pd.concat([fake_df, true_df], ignore_index=True)

data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

def preprocess_text(text):
    return text.lower().strip()

data['text'] = data['text'].apply(preprocess_text)

# Step 2: Tokenize text using BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_length = 128
def tokenize_texts(texts):
    return bert_tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

tokens = tokenize_texts(data['text'])

# Step 3: Convert tokens to NumPy arrays
input_ids = tokens['input_ids'].numpy()
attention_mask = tokens['attention_mask'].numpy()
labels = data['label'].values

# Ensure all lengths match
assert len(input_ids) == len(attention_mask) == len(labels), "Input features and labels must have the same length."

# Train-test split for input IDs and labels
X_train_ids, X_test_ids, y_train, y_test = train_test_split(
    input_ids,
    labels,
    test_size=0.2,
    random_state=42
)

# Train-test split for attention masks
X_train_attention, X_test_attention = train_test_split(
    attention_mask,
    test_size=0.2,
    random_state=42
)

# Sanity checks
assert len(X_train_ids) == len(X_train_attention) == len(y_train), "Training set dimensions must match."
assert len(X_test_ids) == len(X_test_attention) == len(y_test), "Test set dimensions must match."

# Step 4: Define BERT + LSTM model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]
lstm_output = Bidirectional(LSTM(128, return_sequences=False))(bert_output)
dropout = Dropout(0.3)(lstm_output)
output = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
# Convert NumPy arrays to TensorFlow tensors explicitly before passing to the model
history = model.fit(
    [tf.convert_to_tensor(X_train_ids), tf.convert_to_tensor(X_train_attention)],  # Convert to tf.Tensor
    y_train,
    validation_data=([tf.convert_to_tensor(X_test_ids), tf.convert_to_tensor(X_test_attention)], y_test),  # Convert to tf.Tensor
    epochs=3,
    batch_size=16
)

# Step 6: Save model
model.save('fake_news_model')

# Step 7: Flask app for predictions
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess and tokenize input
    tokenized = tokenize_texts([preprocess_text(text)])

    # Convert tokenized inputs to tf.Tensor before passing them to the model
    input_ids_tensor = tf.convert_to_tensor(tokenized['input_ids'])
    attention_mask_tensor = tf.convert_to_tensor(tokenized['attention_mask'])

    # Make prediction
    prediction = model.predict([
        input_ids_tensor,
        attention_mask_tensor
    ])

    result = 'Real' if prediction[0][0] > 0.5 else 'Fake'
    confidence = prediction[0][0] if result == 'Real' else 1 - prediction[0][0]

    return jsonify({
        'result': result,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
