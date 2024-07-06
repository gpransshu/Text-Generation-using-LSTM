

import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F



class Dictionary(object):
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



class Corpus(object):
    
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        #Create a 1-D tensor which contains index of all the words in the file with the help of word2idx
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        # no of required batches            
        num_batches = ids.shape[0] // batch_size     
        #Remove the remainder from the last batch , so that always batch size is constant
        ids = ids[:num_batches*batch_size]
        # return (batch_size,num_batches)
        ids = ids.view(batch_size, -1)
        return ids


# ### Setting the parameter values



embed_size = 128    # Embedding layer size , input to the LSTM
hidden_size = 1024  # Hidden size of LSTM units
num_layers = 1      # no LSTMs stacked
num_epochs = 10     # total no of epochs
batch_size = 20     # batch size
seq_length = 100     # sequence length
learning_rate = 0.002 # learning rate



corpus = Corpus()


import re

# Define the input and output file paths
input_file = 'bg.txt'
output_file = 'data.txt'

# Read the input file
with open(input_file, 'r', encoding='utf-8') as f:
    file = f.read()

# Remove punctuation and convert to lowercase
file = re.sub(r'[^A-Za-z0-9\s]', '', file)
file = file.lower()

# Write the processed text to the output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(file)


ids = corpus.get_data('data.txt',batch_size)


print(ids)


# ids tensors contain all the index of each words
print(ids.shape)


ids


# What is the vocabulary size ?
vocab_size = len(corpus.dictionary)
print(vocab_size)



num_batches = ids.shape[1] // seq_length
print(num_batches)




class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # maps words to feature vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # LSTM layer
        self.linear = nn.Linear(hidden_size, vocab_size) # Fully connected layer

    def forward(self, x, h):
        # Perform Word Embedding 
        x = self.embed(x)

        out, (h, c) = self.lstm(x, h) # (input , hidden state)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)



model = LSTM(vocab_size, embed_size, hidden_size, num_layers)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




# to Detach the Hidden and Cell states from previous history
def detach(states):
    return [state.detach() for state in states]



for epoch in range(num_epochs):
    # initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        
        #move with seq length from the the starting index and move till - (ids.size(1) - seq_length)
        
        # prepare mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length] # fetch words for one seq length  
        targets = ids[:, (i+1):(i+1)+seq_length] # shifted by one word from inputs
        
        states = detach(states)

        outputs,states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
         
        #The gradients are clipped in the range [-clip_value, clip_value]. This is to prevent the exploding gradient problem
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
              
        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



user_input = input("Enter a starting word: ").strip().lower()


with torch.no_grad():
    with open('results.txt', 'w') as f:
        # Initialize hidden and cell states
        state_h = torch.zeros(num_layers, 1, hidden_size)
        state_c = torch.zeros(num_layers, 1, hidden_size)
        state = (state_h, state_c)
        
        # Convert user input to index in the vocabulary
        if user_input in corpus.dictionary.word2idx:
            input_id = np.array([[corpus.dictionary.word2idx[user_input]]], dtype=np.int64)
        else:
            print(f"Word '{user_input}' not found in the dictionary.")
            exit()
        
        abc = ''

        # Loop for generating 500 words
        for i in range(500):
            # Model prediction
            output, state = model(torch.tensor(input_id), state)

            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(output, dim=-1)

            # Sample a word id based on the probabilities
            word_id = np.random.choice(np.arange(vocab_size), p=probabilities.squeeze().numpy())

            # Replace the input_id with sampled word id for the next time step
            input_id = np.array([[word_id]], dtype=np.int64)

            # Get the sampled word and append to generated_words
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '

            # Write the generated word to the results file
            f.write(word)

            abc = abc + word

            # Print progress message every 100 iterations
            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and saved to {}'.format(i + 1, 500, 'results.txt'))


print(abc)
