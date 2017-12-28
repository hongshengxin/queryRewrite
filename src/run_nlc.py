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
import json
import kenlm 
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os
import logging
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from gensim.models.word2vec import Word2Vec
# import tensorflow.python.debug as tf_debug
logging.basicConfig(level=logging.INFO)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
URL = {'model_url':'/home/peng.qiu/QueryRewrite/dataset/one_word_model_size150_window7_sg0_hs0_mincount100.bin',
'data_csv':'/home/peng.qiu/QueryRewrite/dataset/E_all_0to1250620.csv',
'X_all':'/home/peng.qiu/QueryRewrite/dataset/query_traing_byword_x_all.npy',
'Y_all':'/home/peng.qiu/QueryRewrite/dataset/query_traing_byword_y_all.npy',
'code_url':'/home/peng.qiu/nlc-master/',
'common_word':'/home/peng.qiu/QueryRewrite/dataset/common_word.txt',
'test_data':'/home/peng.qiu/QueryRewrite/dataset/test_set_sighan.csv',
'test_X':'/home/peng.qiu/QueryRewrite/dataset/test_X.npy',
'test_Y':'/home/peng.qiu/QueryRewrite/dataset/test_Y.npy',
'lmfile':'/home/peng.qiu/QueryRewrite/model/news_lm.arpa',
}

para = {'learning_rate':0.0003,
'learning_rate_decay_factor':0.95,
'decay_iter':6000,
'max_gradient_norm':10.0,
'dropout':0.1,
'batch_size':1024,
'epochs':40,
'size':150,
'num_layers':3,
'max_seq_len':25,
"optimizer":"rmsprop",
"print_every":10,
"data_dir":"/tmp",
"train_dir":"/home/peng.qiu/nlc-master/train_dir/12_27/",
"tokenizer":"CHAR",
'alpha':0.3,
'beam_size':5,
'pretrain_embed':True,
'add_layer_size':150, 
'singledirec':False,
}
para_info = "numberlayer{8}init_lr{0}_decay_iter{1}_maxgrad{2}_pred_embed{3}_size{4}_batch_size{5}_gpu{6}_dropout{7}_maxseqlen{9}_singledirec{10}".format(para['learning_rate'],para['decay_iter'],para['max_gradient_norm'],para['pretrain_embed'],para['size'],para['batch_size'],os.environ["CUDA_VISIBLE_DEVICES"],para['dropout'],para['num_layers'],para['max_seq_len'],para['singledirec'])
para['train_dir'] = para['train_dir']+para_info+'/'
# para["train_dir"] = "/home/peng.qiu/nlc-master/train_dir/12_27/test/"
print(para_info)
print('add_layer_size')
exec(open(URL['code_url']+'nlc_model_new.py','r').read())
exec(open(URL['code_url']+'util.py','r').read())
exec(open(URL['code_url']+'train.py','r').read())
# exec(open(URL['code_url']+'test.py','r').read())
# exec(open(URL['code_url']+'train_debug.py','r').read())
# exec(open(URL['code_url']+'nlc_debug.py','r').read())
# exec(open(URL['code_url']+'decode.py','r').read())
# exec(open(URL['code_url']+'nlc_my_test.py','r').read())
print(para_info)