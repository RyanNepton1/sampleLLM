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