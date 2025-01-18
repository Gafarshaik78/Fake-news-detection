import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Step 1: Load and preprocess datasets
fake_df = pd.read_csv('/workspaces/Fake-news-detection/data/Fake.csv')
true_df = pd.read_csv('/workspaces/Fake-news-detection/data/True.csv')

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

# step 4 Define BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Custom Keras layer to wrap BERT model
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = bert_model

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state  # or choose `output.pooler_output` if you want pooled embeddings

# Inputs
input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

# Apply the BERT layer
bert_output = BertLayer(bert_model)([input_ids, attention_mask])

# Add LSTM and Dense layers
lstm_output = Bidirectional(LSTM(128, return_sequences=False))(bert_output)
dropout = Dropout(0.3)(lstm_output)
output = Dense(1, activation='sigmoid')(dropout)

# Build and compile the model
model = Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
# Convert NumPy arrays to TensorFlow tensors explicitly before passing to the model
X_train_ids_tensor = tf.convert_to_tensor(X_train_ids)
X_train_attention_tensor = tf.convert_to_tensor(X_train_attention)
y_train_tensor = tf.convert_to_tensor(y_train)

X_test_ids_tensor = tf.convert_to_tensor(X_test_ids)
X_test_attention_tensor = tf.convert_to_tensor(X_test_attention)
y_test_tensor = tf.convert_to_tensor(y_test)

# Enable profiling during training
with tf.profiler.experimental.Profile('logdir'):
    history = model.fit(
        [X_train_ids_tensor, X_train_attention_tensor],
        y_train_tensor,
        validation_data=(
            [X_test_ids_tensor, X_test_attention_tensor],
            y_test_tensor
        ),
        epochs=3,
        batch_size=16
    )


# Step 6: Save model
# Save the model using TensorFlow's save method
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

    # Ensure proper batch dimension for prediction
    input_ids_tensor = tf.expand_dims(input_ids_tensor[0], axis=0)
    attention_mask_tensor = tf.expand_dims(attention_mask_tensor[0], axis=0)

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
