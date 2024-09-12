import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/fine_tuned_bert_amazon/')
model = AutoModelForSequenceClassification.from_pretrained('models/fine_tuned_bert_amazon/')


def fetch_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]
    return reviews


def preprocess_reviews(reviews):
    processed_reviews = [re.sub(r'[^\w\s]', '', review.lower()) for review in reviews]
    return processed_reviews


def batch_sentiment_analysis(reviews):
    sentiments = []
    for review in reviews:
        inputs = tokenizer.encode(review[:512], return_tensors='pt', truncation=True)
        with torch.no_grad():
            result = model(inputs)
        sentiment = int(torch.argmax(result.logits)) + 1
        sentiments.append(sentiment)
    return sentiments


# Fetch and preprocess reviews
url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
reviews = fetch_reviews(url)
reviews = preprocess_reviews(reviews)

# Perform sentiment analysis
sentiments = batch_sentiment_analysis(reviews)

# Create DataFrame
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Visualize Sentiment Distribution
plt.figure(figsize=(8, 6))
df['sentiment'].value_counts().sort_index().plot(kind='bar')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Reviews')
plt.show()

# LIME for interpretability
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])


def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    return probabilities.detach().numpy()


# Explain the prediction for the first review
explanation = explainer.explain_instance(df['review'][0], predict_proba, num_features=10)
explanation.show_in_notebook(text=True)
