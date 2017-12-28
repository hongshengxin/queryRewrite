# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# print('validate')
def create_model(session):
  model = NLCModel(
      para["size"], para["num_layers"], para["max_gradient_norm"], para["batch_size"],
      para["learning_rate"], para["learning_rate_decay_factor"], para["dropout"],add_layer_size = para['add_layer_size'],
      forward_only=False, optimizer=para["optimizer"],pretrain_embed = para['pretrain_embed'],singledirec = para['singledirec'])
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
  for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, para["batch_size"], para["num_layers"]):
    cost,accuracy = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
    valid_costs.append(cost * target_mask.shape[1])
    valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_accuracy.append(accuracy)
  valid_cost = sum(valid_costs) / float(sum(valid_lengths))
  valid_accuracy = sum(valid_accuracy)/len(valid_accuracy)
  return valid_cost,valid_accuracy
  
def train_multi_model(paralist,gpulist):
  assert(len(paralist)==len(gpulist))
  for i in range(len(paralist)):
    para = paralist[i]
    gpu = gpulist[i]



if True:
  """Train a translation model using NLC data."""
  # Prepare NLC data.
  logging.info("Preparing NLC data")

  x_train, y_train, x_dev, y_dev = prepare_nlc_data()

  # save log 
  if not os.path.exists(para["train_dir"]):
    os.makedirs(para["train_dir"])
  file_handler = logging.FileHandler("{0}/log.txt".format(para["train_dir"]))
  logging.getLogger().addHandler(file_handler)

  # save para
  para_file = para['train_dir']+'para.txt'
  para_file = open(para_file,'w')
  for key,value in para.items():
    para_file.write(key)
    para_file.write(' ')
    para_file.write(str(value))
  para_file.close()


  with tf.Session(config = config) as sess:
    model = create_model(sess)
    logging.info('begin to compute')
    initial_cost,intial_accuracy = validate(model, sess, x_dev, y_dev)
    logging.info('Initial validation cost: %f validation accuracy%f' % (initial_cost,intial_accuracy))

    if True:
      tic = time.time()
      params = tf.trainable_variables()
      num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
      toc = time.time()
      print ("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    epoch = 0
    best_epoch = 0
    previous_losses = []
    exp_cost = None
    exp_length = None
    exp_norm = None
    total_iters = 0
    start_time = time.time()
    while (para["epochs"] == 0 or epoch < para["epochs"]):
      epoch += 1
      current_step = 0

      ## Train
      epoch_tic = time.time()
      for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, para["batch_size"], para["num_layers"]):
        # Get a batch and make a step.
        tic = time.time()

        a = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)
        print(a)