#! /usr/bin/python3

#based on Hands on Sci-kit and Tensorflow...

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import os

    import horovod.tensorflow as hvd

    hvd.init() #horovod

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    X_train, y_train=X_train[hvd.rank()::hvd.size()], y_train[hvd.rank()::hvd.size()]
    X_test, y_test=X_test[hvd.rank()::hvd.size()], y_test[hvd.rank()::hvd.size()]

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.intra_op_parallelism_threads=12
    config.inter_op_parallelism_threads=2
    config.allow_soft_placement=True
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    os.environ["OMP_NUM_THREADS"]= "12"


    n_inputs = 28*28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    from tensorflow.contrib.layers import fully_connected

    with tf.name_scope("dnn"):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
        no_op = tf.no_op(name="no_op")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = hvd.DistributedOptimizer(optimizer)
        training_op = optimizer.minimize(loss, name='optimize')

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    bcast = hvd.broadcast_global_variables(0)


    n_epochs = 20
    batch_size = 100

    def shuffle_batch(X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    with tf.Session() as sess:
        init.run()
        bcast.run()
        import time
        start = time.time()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print("Rank: ", hvd.rank(), " epoch: ", epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
        stop = time.time()
        print ("Time: ", stop - start)
