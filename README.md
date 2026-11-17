# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the Iris dataset using a suitable library.

### STEP 2: 

Preprocess the data by handling missing values and normalizing features.

### STEP 3: 

Split the dataset into training and testing sets.

### STEP 4: 

Train a classification model using the training data.

### STEP 5: 

Evaluate the model on the test data and calculate accuracy.

### STEP 6: 

Display the test accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: AKSHAY KARTHICK ASR

### Register Number: 212224230015

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("First 5 rows of dataset: \n", df.head())
print("\nLast 5 rows of dataset:\n", df.tail())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class IrisClassifier(nn.Module):
    def __init__(self, input_size, h1, h2, output_size):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

input_size = X_train.shape[1]
output_size = len(iris.target_names)
h1 = 10
h2 = 11

model = IrisClassifier(input_size=input_size, h1=h1, h2=h2, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
train_model(model, train_loader, criterion, optimizer, epochs)

model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=iris.target_names)


print(f'Test Accuracy: {accuracy:.2f}%\n')
print("Classification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
sample_input = X_test[5].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = iris.target_names[predicted_class_index]

print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')
```

### Dataset Information
<img width="657" height="685" alt="Screenshot 2026-11-18 at 4 07 54 AM" src="https://github.com/user-attachments/assets/9594628f-8336-4646-93bb-99e4a86c365f" />

### OUTPUT

## Confusion Matrix
<img width="522" height="535" alt="Screenshot 2026-11-18 at 4 08 34 AM" src="https://github.com/user-attachments/assets/c132b689-9f0f-4f1e-969a-88f1ba41b79a" />


## Classification Report
<img width="438" height="187" alt="Screenshot 2026-11-18 at 4 08 48 AM" src="https://github.com/user-attachments/assets/e86f2134-a9af-4730-8f1e-0922072f9621" />

### New Sample Data Prediction
<img width="323" height="44" alt="Screenshot 2026-11-18 at 4 09 16 AM" src="https://github.com/user-attachments/assets/5c93b0da-47c5-4333-9baf-584c69663e46" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch
