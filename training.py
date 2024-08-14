import json
from nltk_utilities import tokenize, stem, bag_of_words
from myModel import neural_network
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 2. Set the Data

# =================================================================
# In this part we want to implement the basic functionalities :
# 1. Load the data
# 2. Preprocess the data
# 3. Create the training data
# =================================================================


with open('data.json', 'r') as f:
    data = json.load(f)
    
all_words = []
tags = []
xy = []


for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag) 
    for patten in intent['patterns']:
        w = tokenize(patten)
        all_words.extend(w)
        xy.append((w, tag))


ignore = ['?', '!', '.', ',']
all_words = sorted([stem(w) for w in all_words if w not in ignore])
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)           
    
    
X_train = np.array(X_train)
y_train = np.array(y_train)

# 3. Setting and training the model using pytorch

class ChatDS(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train 
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# ================== Hyperparameters ==================
batch_size = 16
hidden_size = 16
output_size = len(tags)
input_size = len(X_train[0])
lr = 0.001
epochs = 1000
# =====================================================

dataset = ChatDS()
trainin_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = neural_network(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 4. Training the model
for epoch in range(epochs):
    for (words, labels) in trainin_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if((epoch + 1 ) % 100 == 0):
        print(f'epoch {epoch + 1}/{epochs}, loss={loss.item()}')        # Display the loss every 100 epochs

# 5. Save the model

save = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "modelData.pth"
torch.save(save, FILE)

