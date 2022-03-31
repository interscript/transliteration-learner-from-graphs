

import pandas as pd
import pickle
import torch
import torch.nn as nn
from timeit import default_timer as timer


from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List


import transformer as tmer
import decoder as dcder


# Mount google drive on google collab
from google.colab import drive
drive.mount('/content/drive')


### Load and preprocess data

def check_farsi(txt):
    for c in list('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'):
        if c in txt:
            return False
    return True

def check_length(txt):
    return True if len(txt) < 100 else False


PATH_DATA = 'data/df_data_2.csv'
df = pd.read_csv(PATH_DATA)

df['transliterated'] = [d[:-1] for d in df['trans']]
df['farsi'] = [d[:-1] for d in df['farsi']]
df['no_farsi'] = [check_farsi(d) and check_length(d)  for d in df['transliterated']]
df = df[df['no_farsi'] == True]
N = df.shape[0]
df = df[:int(N/2)]


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SRC_LANGUAGE = 'farsi'
TGT_LANGUAGE = 'transliterated'

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Place-holders
token_transform = {}
vocab_transform = {}


token_transform[SRC_LANGUAGE] = lambda txt: dcder.tokenizer(txt)
token_transform[TGT_LANGUAGE] = lambda txt: dcder.tokenizer(txt)


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = list(zip(df[SRC_LANGUAGE],
                          df[TGT_LANGUAGE]))

    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


# code to store load vocab and transform data

"""
import pickle

with open('drive/MyDrive/Transformer/vocab_transform.pickle', 'wb') as handle:
    pickle.dump(vocab_transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('drive/MyDrive/Transformer/vocab_transform.pickle', 'rb') as handle:
    vocab_transform = pickle.load(handle)

""";

# Build Model

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = tmer.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# load model
"""
PATH = "drive/MyDrive/models/model_basic_epoch_1.pt" # jair

transformer.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
transformer.eval();
""";


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = dcder.sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               dcder.tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = list(zip(df[SRC_LANGUAGE], df[TGT_LANGUAGE]))[:N]
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm.tqdm(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = list(zip(df[SRC_LANGUAGE], df[TGT_LANGUAGE]))[-100000:]
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE,
                                collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


# Train Model

NUM_EPOCHS = 20

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch},            Train loss: {train_loss:.3f},            Val loss: {val_loss:.3f},            "f"Epoch time = {(end_time - start_time):.3f}s"))

    df_test = pd.read_csv("drive/MyDrive/Transformer/data/test.csv")

    df_test["trans0_9"] = [translate(transformer, d) for d in df_test["orig"]]
    #df_test.to_csv('drive/MyDrive/test_data/df_test_'+str(epoch)+'.csv')
    ids = evaluation(df_test["trans"], df_test["trans0_9"], df_test["orig"])

    print('save model:::::')
    torch.save(transformer.state_dict(),
               'drive/MyDrive/models/model_basic_epoch_'+str(epoch)+'.pt')
    print('saved model:::::')
    """
    for src, tgt in list(zip(df[SRC_LANGUAGE], df[TGT_LANGUAGE]))[:10]:
        dico = {'src': src, 'tgt': tgt, 'tra': translate(transformer, src)}
        print('SRC::  ', dico['src'])
        print('TGT::  ', dico['tgt'])
        print('TRA::  ', dico['tra'])
        print('')

    print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
    """

    """
    f_1 = open("drive/MyDrive/data/test.txt", "r")
    data_test = f_1.readlines()
    f_1.close()

    f_2 = open('results/test_'+str(epoch)+'.txt', "w")
    for d in data_test:
        f_2.write(d + "\n")
        f_2.write(translate(transformer, d) + "\n")
        f_2.write("\n")
    f_2.close()
    """;
