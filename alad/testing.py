def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logger = logging.getLogger("ALAD.run.{}.{}".format(
    dataset, label))

    # Import model and data
    network = importlib.import_module('alad.{}_utilities'.format(dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.999

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Placeholders
    x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(),
                          name="input_x")
    z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim],
                          name="input_z")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    logger.info('Data loading...')
    trainx, trainy = data.get_train(label)
    if enable_early_stop: validx, validy = data.get_valid(label)
    trainx_copy = trainx.copy()
    testx, testy = data.get_test(label)

    rng = np.random.RandomState(random_seed)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    
    with sv.managed_session(config=config) as sess:
        scores_ch = []
        scores_l1 = []
        scores_l2 = []
        scores_fm = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):

            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_test_time_batch = time.time()

            feed_dict = {x_pl: testx[ran_from:ran_to],
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl:False}

            scores_ch += sess.run(score_ch, feed_dict=feed_dict).tolist()
            scores_l1 += sess.run(score_l1, feed_dict=feed_dict).tolist()
            scores_l2 += sess.run(score_l2, feed_dict=feed_dict).tolist()
            scores_fm += sess.run(score_fm, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_test_time_batch)


        inference_time = np.mean(inference_time)
        logger.info('Testing : mean inference time is %.4f' % (inference_time))

        if testx.shape[0] % batch_size != 0:

            batch, size = batch_fill(testx, batch_size)
            feed_dict = {x_pl: batch,
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl: False}

            bscores_ch = sess.run(score_ch,feed_dict=feed_dict).tolist()
            bscores_l1 = sess.run(score_l1,feed_dict=feed_dict).tolist()
            bscores_l2 = sess.run(score_l2,feed_dict=feed_dict).tolist()
            bscores_fm = sess.run(score_fm,feed_dict=feed_dict).tolist()


            scores_ch += bscores_ch[:size]
            scores_l1 += bscores_l1[:size]
            scores_l2 += bscores_l2[:size]
            scores_fm += bscores_fm[:size]

        model = 'alad_sn{}_dzz{}'.format(do_spectral_norm, allow_zz)
        save_results(scores_ch, testy, model, dataset, 'ch',
                     'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        save_results(scores_l1, testy, model, dataset, 'l1',
                     'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        save_results(scores_l2, testy, model, dataset, 'l2',
                     'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        save_results(scores_fm, testy, model, dataset, 'fm',
                     'dzzenabled{}'.format(allow_zz), label, random_seed, step)