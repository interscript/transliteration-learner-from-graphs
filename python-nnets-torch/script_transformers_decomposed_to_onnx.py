
import pickle

import torch
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence


import transformer as tfmer 
import decoder as dcder


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Languages tags
SRC_LANGUAGE = 'farsi'
TGT_LANGUAGE = 'transliterated'

# Load vocabulary data
with open('../resources/vocab_transform.pickle', 'rb') as handle:
    vocab_transform = pickle.load(handle)
    
# NNets params
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# init Transformer
transformer = tfmer.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)


# Load model
MODEL_PATH = "../resources/model_trained_transformer.pt" 
transformer.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device('cpu')))
transformer.eval();


# Transforms dict
token_transform = {}
token_transform[SRC_LANGUAGE] = lambda txt: dcder.tokenizer(txt) 
token_transform[TGT_LANGUAGE] = lambda txt: dcder.tokenizer(txt) 

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = dcder.sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               dcder.tensor_transform) # Add BOS/EOS and create tensor

    
### Decipher example

src_sentence = 'باز آ باز آ هر آنچه هستی باز آ'

deciphered = dcder.translate(transformer, text_transform, vocab_transform, src_sentence)

# basic test
assert type(deciphered) == str


### Export token_embbedding

batch_size = 16
source_length = 8 
target_length = 8

input_vocab_size = len(vocab_transform[SRC_LANGUAGE])
src = torch.randint(0, input_vocab_size, (source_length, batch_size))


# Export the model
torch.onnx.export(transformer.src_tok_emb, 
                  (src),
                  "../resources/token_src_embbedding.onnx",
                  input_names=['src'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'src': {0: 'source_length', 1: 'batch_size'}},
                  verbose=True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimizatio
                )  

# Export the model
tgt = src
torch.onnx.export(transformer.tgt_tok_emb, 
                  (tgt),
                  "../resources/token_tgt_embbedding.onnx",
                  input_names=['tgt'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'tgt': {0: 'source_length', 1: 'batch_size'}},
                  verbose=True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimizatio
                )  



### Export Positional Encoding

tokens =  transformer.src_tok_emb(src)

torch.onnx.export(transformer.positional_encoding, # transformer.src_tok_emb, 
                  (tokens),
                  "../resources/token_embbedding.onnx",
                  input_names=['tokens'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'tokens': {0: 'source_length', 1: 'batch_size'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimizatio
                )


### Export Generator

src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
num_tokens = src.shape[0]
src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
memory = transformer.encode(src, src_mask)
memory = memory.to(DEVICE)

start_symbol=BOS_IDX

ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
sz = ys.size(0)
tgt = transformer.positional_encoding(
                          transformer.tgt_tok_emb(ys))

tgt_mask = (dcder.generate_square_subsequent_mask(ys.size(0))
                .type(torch.bool)).to(DEVICE)

out = transformer.transformer.decoder(tgt, memory, tgt_mask)
out = out.transpose(0, 1)
outs = out[:, -1]

#prob = transformer.generator(out[:, -1])
#_, next_word = torch.max(prob, dim=1)
#next_word

torch.onnx.export(transformer.generator,
                  (outs),
                  "../resources/transformer_generator.onnx",
                  input_names=['outs'],
                  output_names=['output'],  # the model's output names
                  # dynamic_axes={'outs': 
                  #               {1: 'vocab_size'}}, # {0: 'source_length', 1: 'batch_size'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False  # whether to execute constant folding for optimizatio
                )


### Encoder

src = torch.randint(0, input_vocab_size, (source_length, batch_size))
src =  transformer.positional_encoding(transformer.src_tok_emb(src))
src_mask = torch.randint(
    0, 2, (source_length, source_length)).bool()

torch.onnx.export(transformer.transformer.encoder, 
                  (src, src_mask),
                  "../resources/transformer_encoder.onnx",
                  input_names=['src', 'src_mask'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'src': {0: 'source_length', 1: 'batch_size'},
                                'src_mask': {0: 'target_length', 1: 'target_length'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimizatio
                )


### Decoder

start_symbol=BOS_IDX
DEVICE = 'cpu'

src_sentence = 'باز آ باز آ هر آنچه هستی باز آ'
src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
num_tokens = src.shape[0]
src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

memory = transformer.encode(src, src_mask)

ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
tgt_mask = (dcder.generate_square_subsequent_mask(ys.size(0))
                .type(torch.bool)).to(DEVICE)
tgt = transformer.positional_encoding(
                          transformer.tgt_tok_emb(ys))
# out = transformer.transformer.decoder(tgt, memory, tgt_mask)

torch.onnx.export(transformer.transformer.decoder, 
                  (tgt, memory, tgt_mask),
                  "../resources/transformer_decoder.onnx",
                  input_names=['tgt', 'memory', 'tgt_mask'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'memory': {0: 'seq_length'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimizatio
                )


# finally export vocabulary
import yaml


v_farsi = vocab_transform['farsi']
v_trans = vocab_transform['transliterated']

d = {'farsi': v_farsi.vocab.get_itos(), 
     'transliterated': v_trans.vocab.get_itos()}

with open('../resources/vocab_transform.yaml', 'w') as outfile:
    yaml.dump(d, outfile) #, default_flow_style=False)
