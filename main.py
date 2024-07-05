import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import re

# Vocabulary class
class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __len__(self):
        return len(self.word2idx)

# TextDataset class
class TextDataset(object):
    def __init__(self):
        self.vocab = Vocab()

    def prepare_data(self, filepath, batch_size=20):
        with open(filepath, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.vocab.add_word(word)  
        # Create a tensor with word indices
        indices = torch.LongTensor(tokens)
        token = 0
        with open(filepath, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    indices[token] = self.vocab.word2idx[word]
                    token += 1
        # Calculate number of batches
        num_batches = indices.shape[0] // batch_size     
        # Ensure batch size consistency
        indices = indices[:num_batches * batch_size]
        # Reshape tensor to (batch_size, num_batches)
        indices = indices.view(batch_size, -1)
        return indices

# Setting the parameter values
embed_dim = 128    
hidden_dim = 1024  
num_layers = 4     
num_epochs = 100    
batch_size = 20    
sequence_length = 100     
learning_rate = 0.001    

dataset = TextDataset()

# Define the input and output file paths
input_filepath = 'raw data.txt'
output_filepath = 'processed_data.txt'

# Read the input file
with open(input_filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove punctuation and convert to lowercase
#content = re.sub(r'[^A-Za-z0-9\s]', '', content)
content = re.sub(r'[^A-Za-z\s]', '', content)
content = content.lower()

# Write the processed text to the output file
with open(output_filepath, 'w', encoding='utf-8') as f:
    f.write(content)

indices = dataset.prepare_data(output_filepath, batch_size)

print(indices)
print(indices.shape)

# Vocabulary size
vocab_size = len(dataset.vocab)
print(vocab_size)

num_batches = indices.shape[1] // sequence_length
print(num_batches)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_dim, vocab_size) 

    def forward(self, x, hidden_state):
        x = self.embedding(x)
        out, hidden_state = self.lstm(x, hidden_state)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.fc(out)
        return out, hidden_state

model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def reset_hidden_state(states):
    return [state.detach() for state in states]

for epoch in range(num_epochs):
    hidden_states = (torch.zeros(num_layers, batch_size, hidden_dim).to(device),
                     torch.zeros(num_layers, batch_size, hidden_dim).to(device))
    
    for i in range(0, indices.size(1) - sequence_length, sequence_length):
        inputs = indices[:, i:i+sequence_length].to(device)
        targets = indices[:, (i+1):(i+1)+sequence_length].to(device)
        
        hidden_states = reset_hidden_state(hidden_states)

        outputs, hidden_states = model(inputs, hidden_states)
        loss = loss_function(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
         
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
              
        step = (i+1) // sequence_length
        if step % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

start_word = input("Enter a starting word: ").strip().lower()

with torch.no_grad():
    hidden_state_h = torch.zeros(num_layers, 1, hidden_dim).to(device)
    hidden_state_c = torch.zeros(num_layers, 1, hidden_dim).to(device)
    hidden_states = (hidden_state_h, hidden_state_c)
    
    if start_word in dataset.vocab.word2idx:
        input_idx = np.array([[dataset.vocab.word2idx[start_word]]], dtype=np.int64)
    else:
        print(f"Word '{start_word}' not found in the dictionary.")
        exit()
    
    generated_text = start_word + " "

    for i in range(200):
        output, hidden_states = model(torch.tensor(input_idx).to(device), hidden_states)

        probabilities = torch.softmax(output, dim=-1)

        word_idx = np.random.choice(np.arange(vocab_size), p=probabilities.squeeze().cpu().numpy())

        input_idx = np.array([[word_idx]], dtype=np.int64)

        word = dataset.vocab.idx2word[word_idx]
        word = '\n' if word == '<eos>' else word + ' '

        generated_text += word

        if (i + 1) % 50 == 0:
            print(f'Generated [{i + 1}/500]')

print(generated_text)
