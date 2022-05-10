
import pickle
import yaml

import torch
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence

# own libs
import transformer as tfmer
import decoder as dcder


with open('../config/params.yml', 'r') as f:
    params = yaml.safe_load(f)
    
ONNX_DIR = params['nnets']['ONNX_DIR']

SRC_CHARS = params['transliteration']['SOURCECHARS']
TGT_CHARS = params['transliteration']['TARGETCHARS']
MAX_STR_LEN = params['transliteration']['max_str_len']

# Define special symbols and indices
special_symbols = params['transliteration']['SPECIALSYMBOLS']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = list(range(len(special_symbols)))

# Languages tags
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_LAN = params['transliteration']['SRC_LAN']
TGT_LAN = params['transliteration']['TGT_LAN']

# Load vocabulary data
vocab_transform_path = params['nnets']['VOCAB_TRAFO']
with open(vocab_transform_path, 'rb') as handle:
    vocab_transform = pickle.load(handle)

# NNets params
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LAN])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LAN])
EMB_SIZE = params['nnets']['EMB_SIZE'] # 512
NHEAD = params['nnets']['NHEAD'] # 8
FFN_HID_DIM = params['nnets']['FFN_HID_DIM'] # 512
BATCH_SIZE = params['nnets']['BATCH_SIZE'] # 128
NUM_ENCODER_LAYERS = params['nnets']['NUM_ENCODER_LAYERS'] # 3
NUM_DECODER_LAYERS = params['nnets']['NUM_DECODER_LAYERS'] # 3

# init Transformer
transformer = tfmer.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                       NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)


# Load model
MODEL_PATH = params['nnets']['MODEL_PATH']
# "../resources/model_trained_transformer.pt"
transformer.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device('cpu')))
transformer.eval();


# Transforms dict
token_transform = {}
token_transform[SRC_LAN] = lambda txt: dcder.tokenizer(txt, SRC_CHARS)
token_transform[TGT_LAN] = lambda txt: dcder.tokenizer(txt, TGT_CHARS)

text_transform = {}
for ln in [SRC_LAN, TGT_LAN]:
    text_transform[ln] = dcder.sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               dcder.tensor_transform) # Add BOS/EOS and create tensor


### Decipher example

src_sentence = 'a'
deciphered = dcder.translate(transformer, text_transform, vocab_transform, SRC_LAN, TGT_LAN, src_sentence)
print('test:')
print('source: ', src_sentence)
print('target: ', deciphered)

# basic test
assert type(deciphered) == str


### Export token_embbedding

batch_size = 1 # arbitrary but since ruby code is running one transliteration at time
source_length = 50 # arbitrary length, medium snippets supported
target_length = 50 # arbitrary length, medium snippets supported

input_vocab_size = len(vocab_transform[SRC_LAN])
src = torch.randint(0, input_vocab_size, (source_length, batch_size))

# Export the Model

print('Export token src embbedding')
torch.onnx.export(transformer.src_tok_emb,
                  (src),
                  ONNX_DIR+"token_src_embbedding.onnx",
                  input_names=['src'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'src': {0: 'source_length', 1: 'batch_size'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimizatio
                )


# Export the Model

print('Export token tgt embbedding')
tgt = src
torch.onnx.export(transformer.tgt_tok_emb,
                  (tgt),
                  ONNX_DIR+"token_tgt_embbedding.onnx",
                  input_names=['tgt'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'tgt': {0: 'source_length', 1: 'batch_size'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimizatio
                )



### Export Positional Encoding

print('Export Positional Encoding')
tokens =  transformer.src_tok_emb(src)

torch.onnx.export(transformer.positional_encoding, # transformer.src_tok_emb,
                  (tokens),
                  ONNX_DIR+"positional_embbedding.onnx",
                  input_names=['tokens'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'tokens': {0: 'source_length'}}, #, 1: 'batch_size'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimizatio
                )


### Export Generator

src = text_transform[SRC_LAN](src_sentence).view(-1, 1)
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

print('Export Generator')
torch.onnx.export(transformer.generator,
                  (outs),
                  ONNX_DIR+"transformer_generator.onnx",
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

print('Export Encoder')
src = torch.randint(0, input_vocab_size, (source_length, batch_size))
src =  transformer.positional_encoding(transformer.src_tok_emb(src))
src_mask = torch.randint(
    0, 2, (source_length, source_length)).bool()

torch.onnx.export(transformer.transformer.encoder,
                  (src, src_mask),
                  ONNX_DIR+"transformer_encoder.onnx",
                  input_names=['src', 'src_mask'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'src': {0: 'source_length', 1: 'batch_size'},
                                'src_mask': {0: 'source_length', 1: 'source_length'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimizatio
                )


### Decoder

print('Export Decoder')
start_symbol=BOS_IDX
DEVICE = 'cpu'

#src_sentence = 'a abcd b d e f g h i j k l'
#src = text_transform[SRC_LAN](src_sentence).view(-1, 1)
src = torch.randint(0, input_vocab_size, (source_length, batch_size))

num_tokens = src.shape[0]
src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

memory = transformer.encode(src, src_mask)

ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
tgt_mask = (dcder.generate_square_subsequent_mask(ys.size(0))
                .type(torch.bool)).to(DEVICE)
tgt = transformer.positional_encoding(
                          transformer.tgt_tok_emb(ys))

# out = transformer.transformer.decoder(tgt, memory, tgt_mask)
tokens = transformer.tgt_tok_emb(ys)

torch.onnx.export(transformer.transformer.decoder,
                  (tgt, memory, tgt_mask),
                  ONNX_DIR+"transformer_decoder.onnx",
                  input_names=['tgt', 'memory', 'tgt_mask'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'tgt': {0: 'source_length'},
                                'memory': {0: 'source_length'}, 
                                'tgt_mask': {0: 'target_length', 1: 'target_length'}},
                  verbose=False,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimizatio
                )


# finally export vocabulary
import yaml

V_SRC = vocab_transform[SRC_LAN]
V_TGT = vocab_transform[TGT_LAN]

d = {SRC_LAN: V_SRC.vocab.get_itos(),
     TGT_LAN: V_TGT.vocab.get_itos()}

print('Write Vocab Transform')
with open(ONNX_DIR+'vocab_transform.yaml', 'w') as outfile:
    yaml.dump(d, outfile) #, default_flow_style=False)
