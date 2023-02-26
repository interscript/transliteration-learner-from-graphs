

from torch import Tensor
import torch
import torch.nn as nn
from typing import Iterable, List
from torch.nn import Transformer
import math


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tokenizer(text, VOCAB):
    text = str(text)
    for s in '.,!;:?':
        text = text.replace(s, ' '+str(s)+' ')
    return ''.join([c for c in text if c in VOCAB]).split()


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# Hardcoded Language parameters
DEVICE = 'cpu'


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module,
              text_transform: dict,
              vocab_transform: dict,
              SRC_LANGUAGE: str,
              TGT_LANGUAGE: str,
              src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(
        list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


############################
#   Algo 
# for word decomposition:
############################

"""
def model_wrd_algo(wrd, d_cnt_vocab, ):
  n_char = len(wrd)
  l_encodings = []
  score = 0 # 1
  model = [d_cnt_vocab.get(wrd, 0), d_cnt_vocab.get(wrd, 0)]
  for i in range(1, n_char):
    w_1, w_2 = wrd[:i], wrd[i:]
    i_1, i_2 = d_w_embedding.get(w_1, False), d_w_embedding.get(w_2, False)
    
    if i_1 and i_2:
      if score < len(w_1) * len(w_2):
        score = score
        model = [i_1, i_2]
        
  return model+[d_cnt_vocab.get(wrd, 0)]

def encode_txt(txt):
  l_wrds = re.split('[ ?.,!:;،]', s.strip())
  src = []
  for w in l_wrds:
    for i in model_wrd_algo(w):
      src.append(i)
  return src
"""

############################
#   Evaluation
############################
import re

def evaluation(trans_orig, trans_model):
    l_bugs = []
    tp, fp = 0, 0
    for i, d in enumerate(zip(trans_orig, trans_model)):
        
        l_orig = [s for s in re.split('[ ?.,!:;]', d[0].strip()) if s != '']
        l_model = [s for s  in re.split('[ ?.,!:;]', d[1].strip()) if s != '']

        correct = True
        for o in zip(l_orig, l_model):
            if o[0] == o[1]:
                tp += 1
            else:
                fp += 1
                correct = False
        if not correct:
            l_bugs.append({"id": i, "trans": d[0], "trans_model": d[1]})

    if tp + fp == 0:
        score = 0.
    else:
        score = tp / (tp + fp)

    print({"accuracy": score})
    return [d["id"] for d in l_bugs]
