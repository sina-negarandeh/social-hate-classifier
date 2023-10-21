# text_model.py
# This module contains the code for embedding the text using DistilBERT

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define a custom dataset class for Persian comments
class PersianCommentsDataset(Dataset):
    def __init__(self, tokenizer, messages, labels):
        self.tokenizer = tokenizer
        self.messages = messages
        self.labels = labels

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        message = self.messages[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(label).long()
        return input_ids, attention_mask, label

# Load the train and test data frames from csv files
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

# Load the DistilBERT tokenizer and model from HuggingFace
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased")

# Set the hyperparameters for training and testing
batch_size = 4
epochs = 3
learning_rate = 2e-5

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Create the train and test datasets and dataloaders
messages = train_df['text']
labels = train_df['hate_speech']
train_dataset = PersianCommentsDataset(tokenizer, messages, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_messages = test_df['text']
test_labels = test_df['hate_speech']
test_dataset = PersianCommentsDataset(tokenizer, test_messages, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Move the model to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model on the train set
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    for input_ids, attention_mask, label in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        labels = label.detach().cpu().numpy().tolist()
        train_preds.extend(preds)
        train_labels.extend(labels)
    train_loss /= len(train_dataloader)
    train_acc = accuracy_score(train_labels, train_preds)
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_acc:.4f}")

# Test the model on the test set and get the predictions
model.eval()
test_preds = []
test_labels = []
for input_ids, attention_mask, label in test_dataloader:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    label = label.to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        labels = label.detach().cpu().numpy().tolist()
        test_preds.extend(preds)
        test_labels.extend(labels)

# Convert the predictions and labels to tensors
test_preds_tensor = torch.tensor(test_preds)
test_labels_tensor = torch.tensor(test_labels)

# Evaluate the model on the test set using accuracy and F-score metrics
test_accuracy = torch.sum(test_preds_tensor == test_labels_tensor) / len(test_labels_tensor)
print(f"Test accuracy: {test_accuracy:.4f}")
precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_tensor, test_preds_tensor, average='macro')
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test fscore: {fscore:.4f}")

# Extract the features from the text embeddings using the base model
base_model = model.base_model

# Get the features for the train set
train_dataset = PersianCommentsDataset(tokenizer, messages, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
features_list = []
base_model.eval()
for input_ids, attention_mask, _ in train_dataloader:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        features = base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        features = torch.mean(features, dim=1)
        features_list.append(features)
features_tensor = torch.cat(features_list, dim=0)
print(f"Train features tensor: {features_tensor} with shape of {np.shape(features_tensor)}")
features_list = features_tensor.tolist()
train_df['text_embeddings'] = features_list

# Get the features for the test set
test_dataset = PersianCommentsDataset(tokenizer, test_messages, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
features_list = []
base_model.eval()
for input_ids, attention_mask, _ in test_dataloader:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        features = base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        features = torch.mean(features, dim=1)
        features_list.append(features)
features_tensor = torch.cat(features_list, dim=0)
print(f"Test features tensor: {features_tensor} with shape of {np.shape(features_tensor)}")
features_list = features_tensor.tolist()
test_df['text_embeddings'] = features_list

# Save the train and test data frames with the text embeddings to csv files
train_df.to_csv('train-final.csv')
test_df.to_csv('test-final.csv')
