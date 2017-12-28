# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import random
import string

import numpy as np
from six.moves import xrange
import tensorflow as tf
import kenlm
lm = None
print("decode 0")
para['num_layers'] = 3
def create_model(session, forward_only):
  model = NLCModel(
      para["size"], para["num_layers"], para["max_gradient_norm"], para["batch_size"],
      para["learning_rate"], para["learning_rate_decay_factor"], para["dropout"],
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(para["train_dir"])
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def n_to_text(List):
  text = []
  for i in List:
    t = model.to_text(i)
    t=t.replace('<GO>','')
    t=t.replace('<PAD>','')
    text.append(t)
  return text




def lm_rank(strs, probs):
  if lm is None:
    return strs[0]
  a = para['alpha']
  lmscores = [lm.score(s)/(1+len(s)) for s in strs]
  probs = [ p / (len(s)+1) for (s, p) in zip(strs, probs) ]
  # division by len
  for (s, p, l) in zip(strs, probs, lmscores):
    print(s, p, l)

  rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
  # get rescore 
  rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
  # rerank from big to little
  generated = strs[rerank[-1]]
  lm_score = lmscores[rerank[-1]]
  nw_score = probs[rerank[-1]]
  score = rescores[rerank[-1]]
  return generated #, score, nw_score, lm_score

#  if lm is None:
#    return strs[0]
#  a = FLAGS.alpha
#  rescores = [(1-a)*p + a*lm.score(s) for (s, p) in zip(strs, probs)]
#  rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x:x[1])]
#  return strs[rerank[-1]]

def decode_beam(model, sess, encoder_output, max_beam_size):
  toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
  return toks.tolist(), probs.tolist()

def fix_sent(model, sess, source_token,source_mask):
  # only one sentence 
  # Encode
  encoder_output = model.encode(sess, source_toks, source_mask)
  # Decode
  beam_toks, probs = decode_beam(model, sess, encoder_output, para['beam_size'])
  # De-tokenize
  beam_strs = n_to_text(beam_toks)
  # Language Model ranking
  best_str = lm_rank(beam_strs, probs)
  # Return
  return best_str

def prepare_nlc_test_data():
  X = np.load(URL['test_X'])
  Y = np.load(URL['test_Y'])
  X,Y = limit_max_len(X,Y)

  x_tokens = X
  y_tokens = Y
  y_tokens = add_sos_eos(y_tokens)
  x_padded, y_padded = padded(x_tokens, para['num_layers']), padded(y_tokens, 0)
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
  
  return source_tokens,source_mask,target_tokens,target_mask
  # return X,Y

# load lm file 
if URL['lmfile'] is not None:
  print("Loading Language model from %s" % URL['lmfile'])
  lm = kenlm.LanguageModel(URL['lmfile'])
print("Preparing NLC data in")

source_tokens,source_mask,target_tokens,target_mask= prepare_nlc_test_data()
# X_test,Y_test = prepare_nlc_test_data()
one_toks = source_tokens[:,1]
one_toks = np.reshape(one_toks,[-1,1])
one_mask = source_mask[:,1]
one_mask = np.reshape(one_mask,[-1,1])

tf.reset_default_graph() 
with tf.Session() as sess:
  print("Creating %d layers of %d units." % (para['num_layers'], para['size']))
  model = create_model(sess,False)
  encoder_output = model.encode(sess, one_toks, one_mask)
  beam_toks, probs = decode_beam(model, sess, encoder_output, para['beam_size'])
  # sentence = ""
  # encoder_output = model.encode(sess, source_tokens, source_mask)
  # beam_toks, probs = decode_beam(model, sess, encoder_output, para['beam_size'])
  # De-tokenize
  # beam_strs = model.to_text(beam_toks)
  # Language Model ranking
  # best_str = lm_rank(beam_strs, probs)
  # for i in 
  # output_sent = fix_sent(model, sess, sent)

    # print("Candidate: ", output_sent)

beam_toks = [[2, 10, 51, 521], [2, 49, 314, 0], [2, 4035, 53, 0], [2, 108, 82, 0], [2, 754, 525, 24]]