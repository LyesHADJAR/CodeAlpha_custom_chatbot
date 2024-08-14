import random
import json
import torch
from myModel import neural_network
from nltk_utilities import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data.json', 'r') as f:
    data = json.load(f)
    
FILE = "modelData.pth"
modelData = torch.load(FILE)

model_state = modelData["model_state"]
input_size = modelData["input_size"]
hidden_size = modelData["hidden_size"]
output_size = modelData["output_size"]
all_words = modelData["all_words"]
tags = modelData["tags"]

model = neural_network(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Hamoud"
print("Salam Sahbi, I am Hamoud, how can I help you? (type 'exit' to stop chatting)")
while True:
    sentence = input("You: ")
    if sentence == "exit":
        break
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)
    
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probabilities = torch.softmax(output, dim=1)[0][predicted.item()]
    if probabilities.item() > 0.75:
        for intent in data["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    
    else:
        print(f"{bot_name}: I did not get it Sahbi, can you please ask me another question?")
