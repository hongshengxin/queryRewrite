# -*- coding: utf-8 -*-
from __future__ import unicode_literals
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
  valid_costs, valid_lengths = [], []
  i = 0
  number = 0
  for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, para["batch_size"], para["num_layers"]):
    cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
    i=i+1
    number +=len(source_mask)
    logging.info('validate cost %d %d %f' % (i,number,cost))
    valid_costs.append(cost * target_mask.shape[1])
    valid_lengths.append(np.sum(target_mask[1:, :]))
  valid_cost = sum(valid_costs) / float(sum(valid_lengths))
  return valid_cost


# def train():
if True:
  """Train a translation model using NLC data."""
  # Prepare NLC data.
  logging.info("Preparing NLC data")

  x_train, y_train, x_dev, y_dev = prepare_nlc_data()

  # vocab, _ = nlc_data.initialize_vocabulary(vocab_path)
  # vocab_size = len(vocab)
  # logging.info("Vocabulary size: %d" % vocab_size)

  if not os.path.exists(para["train_dir"]):
    os.makedirs(para["train_dir"])
  file_handler = logging.FileHandler("{0}/log.txt".format(para["train_dir"]))
  logging.getLogger().addHandler(file_handler)

  # print(vars(FLAGS))
  # with open(os.path.join(para["train_dir, "flags.json"), 'w') as fout:
  #   json.dump(para["__flags, fout)
  x_train_test = x_train[:para['batch_size']*10]
  y_train_test = y_train[:para['batch_size']*10]
  
  x_dev_test = x_dev[:para['batch_size']*10]
  y_dev_test = y_dev[:para['batch_size']*10]
  with tf.Session(config = config) as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    model = create_model(sess, False)
    logging.info('begin to compute')
    logging.info('Initial validation cost: %f' % validate(model, sess, x_dev_test, y_dev_test))
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
      for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train_test, y_train_test, para["batch_size"], para["num_layers"]):
        # Get a batch and make a step.
        tic = time.time()

        grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

        toc = time.time()
        iter_time = toc - tic
        total_iters += np.sum(target_mask)
        tps = total_iters / (time.time() - start_time)
        current_step += 1

        lengths = np.sum(target_mask, axis=0)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        print(iter_time)
        print(grad_norm)
        print(cost)
        if not exp_cost:
          exp_cost = cost
          exp_length = mean_length
          exp_norm = grad_norm
        else:
          exp_cost = 0.99*exp_cost + 0.01*cost
          exp_length = 0.99*exp_length + 0.01*mean_length
          exp_norm = 0.99*exp_norm + 0.01*grad_norm

        cost = cost / mean_length

        if current_step % para["print_every"] == 0:
          logging.info('epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, tps %f, length mean/std %f/%f' %
                (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, tps, mean_length, std_length))
      epoch_toc = time.time()

      ## Checkpoint
      checkpoint_path = os.path.join(para["train_dir"], "best.ckpt")

      ## Validate
      valid_cost = validate(model, sess, x_dev_test, y_dev_test)

      logging.info("Epoch %d Validation cost: %f time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))

      if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
        logging.info("Annealing learning rate by %f" % para["learning_rate_decay_factor"])
        sess.run(model.learning_rate_decay_op)
        model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
      else:
        previous_losses.append(valid_cost)
        best_epoch = epoch
        model.saver.save(sess, checkpoint_path, global_step=epoch)
      sys.stdout.flush()