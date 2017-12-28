# -*- coding: utf-8 -*-
from __future__ import unicode_literals
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
batch_size = para['batch_size']
num_layers = para['num_layers']
def pair_iter(X, Y, batch_size, num_layers, num_gpu = None):
  # fdx, fdy = open(fnamex), open(fnamey)
  batches = (len(X) + batch_size - 1)//batch_size

  # while True:
  if num_gpu==None:
    for i in range(batches-1):
      x_tokens = X[i*batch_size:(i+1)*batch_size]
      y_tokens = Y[i*batch_size:(i+1)*batch_size]
      y_tokens = add_sos_eos(y_tokens)
      x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 0)

      source_tokens = np.array(x_padded).T
      source_mask = (source_tokens != PAD_ID).astype(np.int32)
      target_tokens = np.array(y_padded).T
      target_mask = (target_tokens != PAD_ID).astype(np.int32)

      # yield (source_tokens, source_mask, target_tokens, target_mask)
      yield (source_tokens, source_mask, target_tokens, target_mask)
  else:
    for i in range(batches//num_gpu):
      x_tokens = X[i*batch_size*num_gpu:(i+1)*batch_size*num_gpu]
      y_tokens = Y[i*batch_size*num_gpu:(i+1)*batch_size*num_gpu]
      y_tokens = add_sos_eos(y_tokens)
      x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 0)

      source_tokens = np.array(x_padded).T
      source_mask = (source_tokens != PAD_ID).astype(np.int32)
      target_tokens = np.array(y_padded).T
      target_mask = (target_tokens != PAD_ID).astype(np.int32)

      # yield (source_tokens, source_mask, target_tokens, target_mask)
      yield (source_tokens, source_mask, target_tokens, target_mask)

def add_sos_eos(tokens):
  T = []
  for token_list in tokens:
    T.append([SOS_ID] + token_list + [EOS_ID])
  return T
  # return map(lambda token_list: [SOS_ID] + token_list + [EOS_ID], tokens)

def padded(tokens, depth):
  '''
  pad tokens to specify lenth with pyramid shape  
  '''
  maxlen = max(map(lambda x: len(x), tokens))
  align = pow(2, depth)
  # align = pow(2, depth -1)
  padlen = maxlen + (align - maxlen) % align
  # return map(lambda token_list: token_list + [nlc_data.PAD_ID] * (padlen - len(token_list)), tokens)
  P = []
  for token_list in tokens:
    P.append(token_list + [PAD_ID] * (padlen - len(token_list)))
  return P
  # return map(lambda token_list: token_list + [PAD_ID] * (padlen - len(token_list)), tokens)

def prepare_nlc_data():
  X = np.load(URL['X_all'])
  Y = np.load(URL['Y_all'])
  X,Y = limit_max_len(X,Y)
  train_size = len(X)-5000
  index = np.random.choice(len(X),size = len(X),replace=False)
  train_index = index[:train_size]
  dev_index = index[train_size:]
  x_train = X[train_index]
  y_train = Y[train_index]
  x_dev = X[dev_index]
  y_dev = Y[dev_index]

  # show data detail 
  i=0
  x_tokens = X[i*batch_size:(i+1)*batch_size]
  y_tokens = Y[i*batch_size:(i+1)*batch_size]
  y_tokens = add_sos_eos(y_tokens)
  x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 0)

  source_tokens = np.array(x_padded).T
  source_mask = (source_tokens != PAD_ID).astype(np.int32)
  target_tokens = np.array(y_padded).T
  target_mask = (target_tokens != PAD_ID).astype(np.int32)
  print("X shape",X.shape)
  print("Y shape",Y.shape)
  print("source_tokens",source_tokens[:,0])
  print("source_mask",source_mask[:,0])
  print("target_tokens",target_tokens[:,0])
  print("target_mask",target_mask[:,0])
  
  return x_train,y_train,x_dev,y_dev 

def limit_max_len(X,Y):
  limit_X = []
  limit_Y = []
  for i in range(len(X)):
    if len(X[i])<=para['max_seq_len']:
      limit_X.append(X[i])
      limit_Y.append(Y[i])
  limit_X = np.asarray(limit_X)
  limit_Y = np.asarray(limit_Y)
  return limit_X,limit_Y



