# social_hate_classifier.py
# This module contains the code for concatenating the text and graph features and using a fully connected neural network to classify hate speech

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the train and test data frames with the text and graph embeddings from csv files
train_df = pd.read_csv('train-final.csv')
test_df = pd.read_csv('test-final.csv')

# Concatenate the text and graph features in a vector of 784 dimensions
train_df['features'] = train_df['text_embeddings'] + train_df['graph_embeddings']
test_df['features'] = test_df['text_embeddings'] + test_df['graph_embeddings']

# Set the hyperparameters for training and testing
batch_size = 64
num_epochs = 20
learning_rate = 0.001

# Define a fully connected neural network with three layers
class AllJointModel(nn.Module):
    def __init__(self):
        super(AllJointModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64) 
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x

model = AllJointModel()

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# Convert the features and labels to tensors
features = torch.tensor(train_df['features'].to_list())
labels = torch.tensor(train_df['hate_speech'].values)
test_features = torch.tensor(test_df['features'].to_list())
test_labels = torch.tensor(test_df['hate_speech'].values)

# Train the model on the train set
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(features), batch_size):
      inputs = features[i:i+batch_size]
      y = labels[i:i+batch_size]

      optimizer.zero_grad()

      outputs = model(inputs)

      loss = criterion(outputs, y)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % (batch_size * 10) == 0: 
          print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
          running_loss = 0.0

print('Finished training')

# Test the model on the test set and get the predictions
test_outputs = model(test_features) 
test_predictions = torch.argmax(test_outputs, dim=-1) 

# Evaluate the model on the test set using accuracy, precision, recall, and F-score metrics
test_accuracy = torch.sum(test_predictions == test_labels) / len(test_labels)
print('Test accuracy: %.3f' % test_accuracy)

precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, test_predictions, average='macro')
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test fscore: {fscore:.4f}")
