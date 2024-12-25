import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Function to load data
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data['content'].tolist(), data['label'].tolist()

def create_model(input_dim, max_sequence_length, output_dim, num_classes):
    model = Sequential()

    # Embedding Layer (Consider using pre-trained GloVe embeddings here)
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_sequence_length))

    # LSTM Layers
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))

    # Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with learning rate tuning
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(csv_file):
    # Load the data
    texts, labels = load_data(csv_file)
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Tokenizer for text preprocessing
    tokenizer = Tokenizer(num_words=10000)  # Limit vocabulary size to 10,000 words
    tokenizer.fit_on_texts(texts)

    # Convert texts to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to ensure uniform input size
    max_sequence_length = 512  # You can adjust this based on your data
    X = pad_sequences(sequences, maxlen=max_sequence_length)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Create the deep learning model
    model = create_model(input_dim=10000, max_sequence_length=max_sequence_length, output_dim=128, num_classes=len(label_encoder.classes_))

    # Add Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Class Weights (if dataset is imbalanced)
    class_weights = {i: 1.0 for i in range(len(label_encoder.classes_))}  # Adjust this if classes are imbalanced

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
              class_weight=class_weights, callbacks=[early_stopping])

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {score[0]}")
    print(f"Test Accuracy: {score[1]}")

    # Save the model and tokenizer
    model.save("DeepLearning/deep_learning_model.keras")
    with open("DeepLearning/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("DeepLearning/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    # Specify your CSV file path here
    csv_file = 'data/data.csv'  # Replace with your CSV path
    train_model(csv_file)
