import re
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Initialize the tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)


class ReviewDataset(Dataset):

    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        inputs = tokenizer(self.reviews[idx], padding='max_length', truncation=True, max_length=512,
                           return_tensors='pt')
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs


def load_amazon_polarity_dataset():
    dataset = load_dataset('amazon_polarity')

    # Prepare dataset for fine-tuning
    train_reviews = dataset['train']['content']
    train_labels = dataset['train']['label']  # 0 for negative, 1 for positive
    train_dataset = ReviewDataset(train_reviews, train_labels)

    return train_dataset


def preprocess_texts(texts):
    return [re.sub(r'[^\w\s]', '', text.lower()) for text in texts]
