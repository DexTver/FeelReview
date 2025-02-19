import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

REVIEWS_FOLDER = "reviews"

EMOJIS = {"negative": "😠", "neutral": "😐", "positive": "😃"}

def analyze_sentiment(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output = model(**tokens)
    sentiment_id = torch.argmax(output.logits, dim=1).item()
    sentiments = ["negative", "neutral", "positive"]
    return sentiments[sentiment_id]

def process_reviews():
    if not os.path.exists(REVIEWS_FOLDER):
        print(f"❌ Папка {REVIEWS_FOLDER} не найдена!")
        return

    print("Анализ отзывов...\n")
    for filename in os.listdir(REVIEWS_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(REVIEWS_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                sentiment = analyze_sentiment(text)
                print(f"📄 Файл: {filename} - Тональность: {sentiment} {EMOJIS[sentiment]}")

if __name__ == "__main__":
    process_reviews()
