# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import other modules
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from lion_pytorch import Lion # using lion optimizer instead of Adam
from transformer import GPTLanguageModel


# device choice
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
with open('../config/params.yml', 'r') as f:
    params = yaml.safe_load(f)

batch_size = params['nnets']['batch_size'] #64 
block_size = params['nnets']['block_size'] # 256 # maximum context length (sentence length) for predictions?
max_iters = params['nnets']['max_iters'] # 5000 # maximum number of iterations for training
eval_interval = params['nnets']['eval_interval'] #500 # evaluate every n iterations
eval_iters = params['nnets']['eval_iters'] # 200 # number of iterations to evaluate on
learning_rate = params['nnets']['learning_rate'] # 3e-4 # learning rate
n_embd = params['nnets']['n_embd'] # 384 # embedding dimension
n_head = params['nnets']['n_head'] # 6 # number of attention heads
n_layer = params['nnets']['n_layer'] # 6 # number of transformer blocks
dropout = params['nnets']['dropout'] # 0.2 # dropout rate


# file parameters
file_name = params['file_parameters']['file_name'] # 'gdrive/MyDrive/out.txt' # path to the comma separated text file 
n_cutoff = params['file_parameters']['n_cutoff'] # 1000000 # number of lines to read from the file (throughing away the rest)
chars_file_path = params['file_parameters']['chars_file_path'] # 'gdrive/MyDrive/chars.txt' # path to the file containing the list of characters
torch_model_path = params['file_parameters']['torch_model_path'] # 'gdrive/MyDrive/transformer_model.pt' # path to the torch model
onnx_model_path = params['file_parameters']['onnx_model_path'] # 'gdrive/MyDrive/transformer_model.onnx' # path to the onnx model

file = open(file_name, 'r')
data = file.readlines()[:n_cutoff]
file.close()

# extract a sorted list of unique characters in text, for both input and output languages
# and without filtering for simplicity
chars = set()
for d in data:
  [chars.add(c) for c in d]

# extract a sorted list of unique characters in text
chars = sorted(list(chars))
# write chars into a file chars_file_path
with open(chars_file_path, 'w') as f:
    f.write(''.join(chars))
f.close()

vocab_size = len(chars)
# mapping from characters to integers
char2int = {ch: i for i, ch in enumerate(chars)}
# mapping from integers to characters  
int2char = {i: ch for i, ch in enumerate(chars)}

def normalise_and_torch(data):
  """Normalise text data and convert to torch tensor."""
  data = data + [0] * (block_size - len(data))
  return torch.tensor(data[:block_size]).to(device)

# encoder: take a string and return a list of integers using lambda
encode = lambda text: normalise_and_torch([char2int[ch] for ch in text])

# decoder: take a list of integers and return a string
decode = lambda ints: ''.join([int2char[i] for i in ints])


# train/test distribution
n_test = int(0.1 * min(n_cutoff, len(data)))
# set random seed
random.seed(42)
# shuffle data
random.shuffle(data)

# process and split data into train and test
x_train = [encode(d.split(',')[0]) for d in tqdm(data[:-n_test])]
y_train = [encode(d.split(',')[1]) for d in tqdm(data[:-n_test])]
x_test = [encode(d.split(',')[0]) for d in tqdm(data[-n_test:])]
y_test = [encode(d.split(',')[1]) for d in tqdm(data[-n_test:])]
del data

def get_batch(split):
    """Get a batch of data for training or validation."""
    x_data, y_data = (x_train, y_train) if split == 'train' else (x_test, y_test)
    ix = torch.randint(len(x_data) - block_size, (batch_size,))
    return torch.stack([x_data[i] for i in ix]), \
        torch.stack([y_data[i] for i in ix]) 

@torch.no_grad()
def estimate_loss():
    """Estimate the loss on the train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        correct, total = 0, 0
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
            preds = torch.argmax(logits, dim=2)
            correct += (preds == Y).sum().item()
            total += (X.shape[1] * batch_size)
        out[split+'_prec'] = float(correct) / total
        out[split+'_loss'] = losses.mean()
    model.train()
    return out


# create model
model = GPTLanguageModel(vocab_size, n_embd, n_layer, n_head, block_size, dropout, device)
model = model.to(device)
# using lion optimizer instead of Adam
optimizer = Lion(model.parameters(), lr = learning_rate) 
# loss function
#criterion = nn.CrossEntropyLoss()
# optimiser instantiation
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# define the training loop
for iter in tqdm(range(max_iters)):
  
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train_loss']:.4f}, val loss {losses['val_loss']:.4f}")
        print(f"step {iter}: train prec {losses['train_prec']:.4f}, val prec {losses['val_prec']:.4f}")

    # sample a batch of data
    x, y = get_batch('train')

    # evaluate the loss
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print('Training complete!')

# save model
torch.save(model.state_dict(), torch_model_path)

