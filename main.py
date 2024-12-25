import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from crawler.crawler import fetch_onion_content
from parser.parser import parse_html
from save_to_csv import save_results_to_csv

def load_resources():
    model = load_model('DeepLearning/deep_learning_model.keras')

    with open('DeepLearning/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('DeepLearning/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

# Function to predict the threat level of the content
def predict_threat(content, model, tokenizer, label_encoder):
    sequence = tokenizer.texts_to_sequences([content])
    padded_sequence = pad_sequences(sequence, maxlen=512)  # Ensure same length as training data

    prediction = model.predict(padded_sequence)
    
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    
    predicted_label = label_encoder.inverse_transform([predicted_label_index])
    
    return predicted_label[0]

# Function to get the onion links from a file
def get_onion_links(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    model, tokenizer, label_encoder = load_resources()

    links = get_onion_links("data/links.txt")
    results = []

    for url in links:
        print(f"[INFO] Crawling: {url}")
        html = fetch_onion_content(url)
        
        if html:
            parsed = parse_html(html)
            threat_level = predict_threat(parsed["content"], model, tokenizer, label_encoder)
            results.append({
                "URL": url,
                "Title": parsed["title"],
                "Threat Level": threat_level
            })

    save_results_to_csv(results)

if __name__ == "__main__":
    main()
