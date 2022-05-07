SRC_LAN
import yaml
import pandas as pd
import pickle
import tqdm
from timeit import default_timer as timer

import torch
import torch.nn as nn

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List

# own libs
import transformer as tmer
import decoder as dcder


"""
# Mount google drive on google collab
 from google.colab import drive
drive.mount('/content/drive')
"""

with open('../config/params.yml', 'r') as f:
    params = yaml.safe_load(f)

SRC_CHARS = params['transliteration']['SOURCECHARS']
TGT_CHARS = params['transliteration']['TARGETCHARS']
MAX_STR_LEN = params['transliteration']['max_str_len']


### Load and preprocess data

def clean_chars(txt, CHARS):
    txt = str(txt)
    return ''.join([c for c in txt if c in CHARS]).strip()


def check_length(txt):
    return True if 0 < len(txt) < MAX_STR_LEN else False


# train data
TRAIN_DATA = params['nnets']['TRAIN_DATA']
df = pd.read_csv(TRAIN_DATA,
                 names=['source', 'transliterated'])

# clean up / normalise data
df['source'] = [clean_chars(d, SRC_CHARS) for d in df['source']]
df['transliterated'] = [clean_chars(d, TGT_CHARS) for d in df['transliterated']]
df['valid'] = [check_length(d[0]) and check_length(d[1]) for d in zip(df['source'], df['transliterated'])]
df = df[df['valid'] == True]
del df['valid']
N = df.shape[0]
print('train data length: ', N)

# test data
TEST_DATA = params['nnets']['TEST_DATA']
df_test = pd.read_csv(TEST_DATA)


### Build model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SRC_LAN = 'source'
TGT_LAN = 'transliterated'


# Make sure the tokens are in order of their indices to properly insert them in vocab
# Define special symbols and indices
special_symbols = params['transliteration']['SPECIALSYMBOLS']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = list(range(len(special_symbols)))

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LAN] = lambda txt: dcder.tokenizer(txt, SRC_CHARS)
token_transform[TGT_LAN] = lambda txt: dcder.tokenizer(txt, TGT_CHARS)

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LAN: 0, TGT_LAN: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


for ln in [SRC_LAN, TGT_LAN]:
    # Training data Iterator
    train_iter = list(zip(df[SRC_LAN],
                          df[TGT_LAN]))

    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LAN, TGT_LAN]:
    vocab_transform[ln].set_default_index(UNK_IDX)


# code to store load vocab and transform data
# on google colab
# with open('drive/MyDrive/Transformer/vocab_transform.pickle', 'wb') as handle:
vocab_transform_path = params['nnets']['VOCAB_TRAFO']
print('print vocab_transform to ', vocab_transform_path)
with open(vocab_transform_path, 'wb') as handle:
    pickle.dump(vocab_transform, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Build Model

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LAN])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LAN])
EMB_SIZE = params['nnets']['EMB_SIZE'] # 512
NHEAD = params['nnets']['NHEAD'] # 8
FFN_HID_DIM = params['nnets']['FFN_HID_DIM'] # 512
BATCH_SIZE = params['nnets']['BATCH_SIZE'] # 128
NUM_ENCODER_LAYERS = params['nnets']['NUM_ENCODER_LAYERS'] # 3
NUM_DECODER_LAYERS = params['nnets']['NUM_DECODER_LAYERS'] # 3

transformer = tmer.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# If load model
"""
PATH = "drive/MyDrive/models/model_basic_epoch_1.pt" # jair
transformer.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
transformer.eval();
""";


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LAN, TGT_LAN]:
    text_transform[ln] = dcder.sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               dcder.tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LAN](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LAN](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = list(zip(df[SRC_LAN], df[TGT_LAN]))[:N]
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm.tqdm(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tmer.create_mask(src, tgt_input)

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

    val_iter = list(zip(df[SRC_LAN], df[TGT_LAN]))[-100000:]
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE,
                                collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tmer.create_mask(src, tgt_input)

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
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


    df_test["translit_model"] = [dcder.translate(transformer, text_transform, vocab_transform, SRC_LAN, TGT_LAN, d) for d in df_test["source"]]

    ids = dcder.evaluation(df_test["translit"], df_test["translit_model"])
    print('errors:', df_test.loc[ids])
    model_save = 'data/model_basic_epoch_'+str(epoch)+'.pt'
    print('save model: '+ model_save )
    torch.save(transformer.state_dict(),
               'data/model_basic_epoch_'+str(epoch)+'.pt')
    # print('model saved:::::')
