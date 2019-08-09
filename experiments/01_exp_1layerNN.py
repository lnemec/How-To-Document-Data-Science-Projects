def experiment_01():
    """
    **Model and Set-up**

    An educational and detailed tutorial addressing the training of a machine
    learning model to solve the MNIST classification Model can be found
    `here`_. We start with a very simple one-layer neural network.

    .. _here: https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

    In this experiment, we use TensorFlow Version 1.13.1

    **Model**

      * linear classifier (1-layer NN)

    **Parameter**

      * image_size = 28 x 28
      * labels_size = 10
      * learning_rate = 0.05
      * steps_number = 1000
      * batch_size = 200

    **Optimiser**

      * tf.train.GradientDescentOptimizer

    **Results**

      * Step 100, training accuracy 90.00, training loss 0.38
      * Step 200, training accuracy 94.50, training loss 0.27
      * Step 300, training accuracy 96.00, training loss 0.21
      * Step 400, training accuracy 97.50, training loss 0.17
      * Step 500, training accuracy 98.50, training loss 0.14
      * Step 600, training accuracy 99.00, training loss 0.12
      * Step 700, training accuracy 99.00, training loss 0.11
      * Step 800, training accuracy 100.00, training loss 0.10
      * Step 900, training accuracy 100.00, training loss 0.09
      * Step 1000, training accuracy 100.00, training loss 0.08

    Test accuracy: 86.44 %
    Validation accuracy: 88.96 %

    """
    import example_mnist.get_mnist_data as em
    import tensorflow as tf
    import numpy as np
    import os
    #
    # Fix for MacOS openMP setup.
    # For details see: https://github.com/dmlc/xgboost/issues/1715
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    #----------------------------------------------------------------------------#
    # Get Dataset
    #----------------------------------------------------------------------------#

    data = em.load()

    #----------------------------------------------------------------------------#
    # Define Model
    #----------------------------------------------------------------------------#

    image_size = data['image_size']
    labels_size = data['labels_size']
    learning_rate = 0.05
    steps_number = 1000
    batch_size = 200


    print("Using TensorFlow", tf.__version__)

    # Define placeholders
    training_data = tf.placeholder(tf.float32, [None, image_size])
    labels = tf.placeholder(tf.float32, [None, labels_size])

    # Model parameters: W and b
    W = tf.get_variable("W", shape=(image_size, labels_size), dtype=tf.float32, trainable=True)
    b = tf.get_variable("b", shape=(labels_size,), dtype=tf.float32)

    # Compute predictions
    output = tf.matmul(training_data, W) + b

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

    # Training step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run the training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # An epoch is finished when we have looked at all training samples
    for epoch in range(steps_number):

        batch_losses = []
        # Get the next batch
        for batch_start in range(0, image_size, batch_size):

            input_batch = data['x_train'][batch_start:batch_start + batch_size]
            labels_batch = data['y_train'][batch_start:batch_start + batch_size]
            feed_dict = {training_data: input_batch, labels: labels_batch}

            _, batch_loss = sess.run([train_step, loss], feed_dict)

            # collect batch losses
            batch_losses.append(batch_loss)
            train_loss = np.mean(batch_losses)

        # ----------------------------------------------------------------------------#
        # Output Results
        # ----------------------------------------------------------------------------#
        # Print the accuracy progress on the batch every 100 steps
        if (epoch+1)%100 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict, session=sess)
            print("Step %i, training accuracy %.2f, training loss %.2f"%(epoch+1, train_accuracy*100, train_loss))

    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: data['x_test'], labels: data['y_test']}, session = sess)
    print()
    print("Test accuracy: %g %%"%(test_accuracy*100))

    # Evaluate on the validation set
    validation_accuracy = accuracy.eval(feed_dict={training_data: data['x_validation'], labels: data['y_validation']}, session = sess)
    print("Validation accuracy: %g %%"%(validation_accuracy*100))

if __name__ == "__main__":
    experiment_01()