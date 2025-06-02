# sample GPT model specific to shakespeare's works

# open the training dataset file and read its content
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#printing some information about the dataset
print("the length of the dataset in characters is:", len(text))
print("the first 1000 characters of the dataset are:", text[:1000])

# print all unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers (charater level encoding)
stoi = { ch: i for i, ch in enumerate(chars) } #encoding
iots = { i: ch for i, ch in enumerate(chars) } #decoding
encode = lambda s: [stoi[c] for c in s] #encoding function
decode = lambda l: ''.join([iots[i] for i in l]) #decoding function

# subword encoder/decoder
print(encode("hello world"))
print(decode(encode("hello world")))

# encode(tokenize) the entire dataset text and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print the shape of the data tensor
print(data.shape, data.dtype)
print(data[:1000])  # print the first 1000 characters now in the form GPT tokens

# now we can spearate the data into a training set and a validation set
n = int(0.9 * len(data))  # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# now for the training loop we need to create a function that generates batches of data
block_size = 8  # how many characters to predict at once
train_data[:block_size+1]

# this fuction shows how the modeis trained to predict everything up to block_size. after that requires trucating
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size): # time dimension
    context = x[:t+1]
    target = y[t]
    print(f"when the input is {context}, the target is {target}")


# traing the batch dimension
torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
#block_size = 8  # what is the maximum context length for predictions? (defined above)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # random starting points for each sequence in the batch
    x = torch.stack([data[i:i+block_size] for i in ix])  # input sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # target sequences
    return x, y # i belive this create a 4 by 8 tensor of inputs and a 4 by 8 tensor of targets
# test the get_batch function
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

# explaination: this 4 by 8 array contains 32 independent examples for the transformer to examine
for b in range(batch_size):# batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, :t+1]  # input sequence up to time t
        target = yb[b, t]  # target at time t
        print(f"when the input is {context.tolist()}, the target: {target}")

# now we can define a simple bigram model
#import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # embedding layer

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C) tensor of logits

        if targets is None:
            loss = None # measure of quality of logits with respect to the logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# example usage of the BigramLanguageModel
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# example of generating text with the model
#idx = torch.zeros((1, 1), dtype=torch.long)  # start with a single token
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32  # increase batch size for better training stability
for steps in range(10000):

    # sample a  batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # zero the gradients
    loss.backward()  # backpropagation
    optimizer.step()  # update the model parameters

    print(loss.item())

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
