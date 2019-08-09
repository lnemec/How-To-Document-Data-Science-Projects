def experiment_02():
    """
    **Model and Set-up**

    An educational and detailed tutorial addressing the training of a machine
    learning model to solve the MNIST classification Model can be found
    `here`_. The one-layer neural network already gave reasonable results. In
    this experiment, we optimise the learning rate by varying the parameter
    between 0.01 and 0.5.

    .. _here: https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

    In this experiment, we use TensorFlow Version 1.13.1

    **Model**

      * linear classifier (1-layer NN)

    **Parameter**

      * image_size = 28 x 28
      * labels_size = 10
      * **learning_rate = [0.5, 0.1, 0.01]**
      * steps_number = 1000
      * batch_size = 200

    **Optimiser**

      * tf.train.GradientDescentOptimizer

    **Results**


    **1. RUN**

    **Training for Learning Rate: 0.50**

      * Step 100, training accuracy 100.00, training loss 0.08
      * Step 200, training accuracy 100.00, training loss 0.04
      * Step 300, training accuracy 100.00, training loss 0.03
      * Step 400, training accuracy 100.00, training loss 0.02
      * Step 500, training accuracy 100.00, training loss 0.02
      * Step 600, training accuracy 100.00, training loss 0.01
      * Step 700, training accuracy 100.00, training loss 0.01
      * Step 800, training accuracy 100.00, training loss 0.01
      * Step 900, training accuracy 100.00, training loss 0.01
      * Step 1000, training accuracy 100.00, training loss 0.01

    Test accuracy: 85.99 %
    Validation accuracy: 88.04 %


    **2. RUN**

    **Training for Learning Rate: 0.10**

      * Step 100, training accuracy 95.00, training loss 0.26
      * Step 200, training accuracy 97.50, training loss 0.17
      * Step 300, training accuracy 98.50, training loss 0.12
      * Step 400, training accuracy 100.00, training loss 0.09
      * Step 500, training accuracy 100.00, training loss 0.08
      * Step 600, training accuracy 100.00, training loss 0.06
      * Step 700, training accuracy 100.00, training loss 0.06
      * Step 800, training accuracy 100.00, training loss 0.05
      * Step 900, training accuracy 100.00, training loss 0.04
      * Step 1000, training accuracy 100.00, training loss 0.04

    Test accuracy: 86.4 %
    Validation accuracy: 88.46 %


    **3. RUN**

    **Training for Learning Rate: 0.01**

      * Step 100, training accuracy 81.50, training loss 0.82
      * Step 200, training accuracy 85.00, training loss 0.58
      * Step 300, training accuracy 88.00, training loss 0.48
      * Step 400, training accuracy 89.50, training loss 0.42
      * Step 500, training accuracy 92.00, training loss 0.38
      * Step 600, training accuracy 92.50, training loss 0.35
      * Step 700, training accuracy 93.00, training loss 0.32
      * Step 800, training accuracy 93.50, training loss 0.30
      * Step 900, training accuracy 94.00, training loss 0.28
      * Step 1000, training accuracy 95.00, training loss 0.26

    Test accuracy: 86.27 %
    Validation accuracy: 88.58 %

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
    learning_rate = [0.5, 0.1, 0.01]
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

    for lr in learning_rate:
        print("Training for Learning Rate: %.2f"%lr)
        # Training step
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

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
    experiment_02()