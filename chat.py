import random
import json
import torch
from nltk_utils import tokenize, bag_of_word
from model import NeuralNet

# define device gpu or cpu
device= torch.device('gpu' if torch.cuda.is_available() else 'cpu')
# Read the data
with open('intents.json', 'r') as f:
    intents= json.load(f)
# load the model
FILE="data.pth"
data= torch.load(FILE)  

#paramters
input_size= data["input_size"]
output_size= data["output_size"]
hidden_size= data["hidden_size"]
all_words= data["all_words"]
tags= data["tags"]
model_state= data["model_state"]
#model
model= NeuralNet(input_size, hidden_size, output_size).to(device)
#model state
model.load_state_dict(model_state)
#model evaluation
model.eval()

# implement the chatbot

bot_name= "Mari"
print("How can i help you? type 'exit' to exit")
def get_response(msg):
    # sentence= input("You: ")
    # if sentence.lower()=="exit":
    #     break
    sentence= tokenize(msg.lower())
    X= bag_of_word(sentence, all_words)
    X= X.reshape(1, X.shape[0])
    X= torch.from_numpy(X)
    
    output= model(X)
    # _, predicted= torch.max(output, dim=1)
    # tag= tags[predicted.item()]
    # # Predicted Probability
    # # probs= torch.softmax(output, dim=1)
    # # prob= probs[predicted.item()]
    # # Directly access the probability value without using an index
    # prob = torch.softmax(output, dim=1).max().item()
    
    # if prob> 0.75:
    #     for intent in intents["intents"]:
    #         if tag== intent["tag"]:
    #             print(f"{bot_name}: {random.choice(intent['responses'])}")
    # else:
    #     print(f"{bot_name}: I don't understand, please try to say another thing")
    _, predicted = torch.max(output, dim=1)

    # Directly access the probability value without using an index
    prob = torch.softmax(output, dim=1).max().item()

    if prob > 0.7:
        predicted_tag = tags[predicted.item()]  # Get the predicted tag
        for intent in intents["intents"]:
            if predicted_tag == intent["tag"]:
              return random.choice(intent['responses'])
    else:
        return "I don't understand, please try saying something else"