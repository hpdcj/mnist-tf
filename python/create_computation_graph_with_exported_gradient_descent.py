#based on Hands on Sci-kit and Tensorflow...

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import os
    import random as rn

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

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

    with tf.name_scope("modify_weights"):
        for layer in ["hidden1/weights", "hidden1/biases",
                      "hidden2/weights", "hidden2/biases",
                      "outputs/weights", "outputs/biases"]:
            with tf.variable_scope("", reuse=True):
                var = tf.get_variable(layer)
                placeholder = tf.placeholder(tf.float32, shape=var.get_shape(), name="p"+layer)
                tf.assign(var, placeholder, name="assign")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = tf.placeholder_with_default(0.01, shape=[], name="learning_rate")

    with tf.name_scope("train_simple"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss, name='optimize')

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        gradients_named = tf.identity_n(gradients, name='compute_gradients')
        gradients_named_length = tf.Variable(len(gradients_named), name='compute_gradients_output_length', trainable=False)
        training_op = optimizer.apply_gradients(gradients, name='apply_gradients')


    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    init = tf.global_variables_initializer()

    with open ('graph.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

    saver = tf.train.Saver()

    n_epochs = 40
    batch_size = 50

    def shuffle_batch(X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

        save_path = saver.save(sess, "./my_model_final.ckpt")