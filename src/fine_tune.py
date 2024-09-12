from datasets import load_dataset

# Load the Amazon Polarity dataset
dataset = load_dataset('amazon_polarity')

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Initialize the tokenizer and model
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Define a dataset class
class ReviewDataset(torch.utils.data.Dataset):
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


# Prepare dataset for fine-tuning
train_reviews = dataset['train']['content']
train_labels = dataset['train']['label']  # 0 for negative, 1 for positive
train_dataset = ReviewDataset(train_reviews, train_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()
model.save_pretrained('models/fine_tuned_bert_amazon/')

