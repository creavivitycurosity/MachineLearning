import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load and prepare data
df = pd.read_csv('data.csv')
text = df['street'] + ', ' + df['city'] + ', ' + df['statezip'] + ', ' + df['country']
text = text.str.lower()

chars = sorted(list(set(''.join(text))))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {int(idx): char for idx, char in enumerate(chars)}  # Convert idx to int

# Prepare sequences
seq_length = 40
sequences = []
next_chars = []

for address in text:
    for i in range(len(address) - seq_length):
        sequences.append([char_to_idx[char] for char in address[i:i+seq_length]])
        next_chars.append(char_to_idx[address[i+seq_length]])

X = torch.LongTensor(sequences)
y = torch.LongTensor(next_chars)

X = torch.LongTensor(X)
y = torch.LongTensor(y)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Initialize the model
model = SimpleRNN(len(chars), 128, len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 50
batch_size = 128

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def generate_text(model, start_string, length):
    generated = start_string.lower()
    for _ in range(length):
        x = torch.LongTensor([[char_to_idx[char] for char in generated[-seq_length:]]])
        prediction = model(x)
        next_char_idx = int(torch.argmax(prediction).item())  # Convert to int
        next_char = idx_to_char[next_char_idx]
        generated += next_char
    return generated

# Interactive loop
print("\nNow you can chat with the model!")
print("Enter a starting string (or 'quit' to exit):")

while True:
    start_string = input("> ")
    if start_string.lower() == 'quit':
        break
    
    generated = generate_text(model, start_string, 50)
    print("Generated text:", generated)

print("Thank you for chatting!")