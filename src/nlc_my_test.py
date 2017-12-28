# -*- coding: utf-8 -*-
from __future__ import unicode_literals
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
  
  # return source_tokens,source_mask,target_tokens,target_mask
  return X,Y

def create_model(session, forward_only):
  model = NLCModel(
      para["size"], para["num_layers"], para["max_gradient_norm"], para["batch_size"],
      para["learning_rate"], para["learning_rate_decay_factor"], para["dropout"],
      forward_only=forward_only, optimizer=para["optimizer"])
  ckpt = tf.train.get_checkpoint_state(para["train_dir"])
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    logging.info("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
  return model


def validate(model, sess, x_dev, y_dev):
  valid_costs, valid_lengths ,valid_accuracy= [], [] ,[]
  i=0
  for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, para["batch_size"], para["num_layers"]):
    cost,accuracy = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
    # i=i+1
    logging.info('validate cost %d %f accuracy%f' % (i,cost,accuracy))
    valid_costs.append(cost * target_mask.shape[1])
    valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_accuracy.append(accuracy)
  valid_cost = sum(valid_costs) / float(sum(valid_lengths))
  valid_accuracy = sum(valid_accuracy)/len(valid_accuracy)
  return valid_cost,valid_accuracy


# def train():
if True:
  """Train a translation model using NLC data."""
  # Prepare NLC data.
  logging.info("Preparing NLC data")
  x_dev,y_dev = prepare_nlc_test_data()
  x_tokens = x_dev
  y_tokens = y_dev
  y_tokens = add_sos_eos(y_tokens)
  x_padded, y_padded = padded(x_tokens, para['num_layers']), padded(y_tokens, 0)
  source_tokens = np.array(x_padded).T
  source_mask = (source_tokens != PAD_ID).astype(np.int32)
  target_tokens = np.array(y_padded).T
  target_mask = (target_tokens != PAD_ID).astype(np.int32)
  with tf.Session(config = config) as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    model = create_model(sess, False)
    logging.info('begin to compute')
    test_cost,test_accuracy = validate(model, sess, x_dev, y_dev)
    cost,accuracy = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
    logging.info('test accuracy: %f' % (accuracy))

