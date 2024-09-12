from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/fine_tuned_bert_amazon/')
model = AutoModelForSequenceClassification.from_pretrained('models/fine_tuned_bert_amazon/')


def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    return probabilities.detach().numpy()


def explain_prediction(text, num_features=10):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    explanation = explainer.explain_instance(text, predict_proba, num_features=num_features)
    explanation.show_in_notebook(text=True)
    return explanation


def plot_explanation(explanation):
    fig = explanation.as_pyplot_figure()
    plt.show()


def explain_and_plot(text):
    explanation = explain_prediction(text)
    plot_explanation(explanation)
