===============================================================================
TensorFlow
===============================================================================

.. contents:: Table of Contents

Introducing Tensor Flow
***********************

TensorFlow Website: https://www.tensorflow.org/

Installation Notes
------------------

- Best to work on python3. If doing so, use IPython3 for it to work correctly.

Possible Imports
----------------

numpy
    Used for scientific computing

math
    Contains math functions

matplotlib
    Allows for plotting and animating data

Training a Model
----------------

Requires:

- Prepared data
- Inference: Function that makes predictions
- Loss Measurement: Way to measure quality of predictions made
- Optimizer to Minimize Loss

House Example Implementation:

- Generated house size and price data (70% to train, 30% to test)
- Price = (sizeFactor * size) + priceOffset
- Mean Square Error
- Gradient Descent Optimizer

Tensor Types
------------

- Constant: constant value
- Variable: values adjusted in graph
- PlaceHolder - used ot pass data into graph

Tensor Properties
-----------------

Tensor
    An n-dimensional array or list used in Tensor to represent all data

1. Rank
    Dimensionality of a Tensor

2. Shape
    Same of data in Tensor. Related to Rank.

3. Type
    The data type contained in the tensor: float32, int8, string, bool, qint8, etc.

Examples::

    Rank 0: Scalar, Ex: 145, Shape: []
    Rank 1: Vector, Ex: v = [1, 3, 2, 5, 7], Shape: [5]
    Rank 2: Matrix, Ex: m = [ [1,5,6], [5,3,4] ], Shape: [2,3]
    Rank 3: 3-Tensor (cube)

Quantitized Values
    Scaled values to reduce size and processing time

Methods
~~~~~~~

- get_shape(): returns shape
- reshape(): changes shape
- rank(): returns rank
- dtype(): return data type
- cast(): change data type

Gradient Descent
----------------

Gradient descent is a popular family of methods for adjusting values to reduce error.
Each step is to be in the direction of the most reduction in loss.

    "Trying to find the fastest way down a hill."

.. note::

    If the learning rate is too high, the process will "bounce" around - not finding the lowest loss.


Creating Neural Networks in TensorFlow
**************************************

Intro to Neural Networks
------------------------

- **Inputs**: Contains values from the data, normally numbers
- **Weights**: Values multiplied by each input that are learned as the model is trained
- **Bias**: Allows for adjustment of the contribution of a specific neuron
- **Sum**: sum(Inputs * Weights) + Bias
- **Activation**: Processes the sum

Forward Propagation
    Neuron sending forward its computed value

Back Propagation
    1. Compute Loss
    2. Optimize to minimize loss

.. code-block:: python
    :caption: Linear Regression Example

    # Weights: size_factor, Bias: price_offset
    tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

    # Compute the loss (Mean Square Error)
    tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

    # Adjusts the values to reduce the loss
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

Simple Neural Network
---------------------

Creating a simple neural network that identifies digits 0-9 from handwritten digits found in the MNIST data set.

MNIST
~~~~~

- http://yann.lecun.com/exdb/mnist
- 70,000 data points
    - 55,000 training
    - 10,000 test
    - 5,000 validation
- 28x28 grayscale image
- Label: 0-9

Implementation
~~~~~~~~~~~~~~

**1. Prepared Data**: MNIST Data

    .. code-block:: python
        :caption: Pull down the data from the MNIST site

        # We use the TF helper function to pull down the data from the MNIST site
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    .. code-block:: python
        :caption: Initialize the placeholder for each image

        # x is placeholder for the 28 X 28 image data
        x = tf.placeholder(tf.float32, shape=[None, 784])

        # The first value is the data type
        # `None` in shape indicates we know that it exists, but we don't know how many items will be in this dimension (# of pictures)
        # `784` in shape is for the 28x28 pixels - Each a float value

    .. code-block:: python
        :caption: Initialize placeholder for the predicted probability of each digit

        # y_ is called "y bar" and is a 10 element vector, containing the predicted probability of each
        #   digit(0-9) class.  Such as [0.14, 0.8, 0,0,0,0,0,0,0, 0.06]
        y_ = tf.placeholder(tf.float32, [None, 10])

        # `None` once again represents the unknown # of pictures

    .. code-block:: python
        :caption: Initialize the weights and biases to zero

        # define weights and balances
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

    ``b`` doesn't need the additional dimension due to `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.

**2. Inference**: sum(x * weight) + bias -> activation

    .. code-block:: python
        :caption: Define the model

        # define our inference model
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # Order matters for the matrix multiplication since it determines the shape
        # SoftMax is the activation function
        # Resulting Tensor has a shape=[None, 10]

    **SoftMax**
        An activation function that is typically used in the output layer when trying to classify what class you have.
        Squashes the values within the tensor to [0,1]

    **Logit**
        the vector of raw (non-normalized) predictions that a classification model generates,
        which is ordinarily then passed to a normalization function.

    **Cross-Entropy**
        A loss function that measures the performance of a classification model whose output is a probability value between 0 and 1.

**3. Loss Measurement**: Cross Entropy

    .. code-block:: python
        :caption: Compare predicted digit ``y`` with actual digit ``y_`` then return the reduced mean

        # loss is cross entropy
        cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # Returns the mean of all the losses between the comparisons

**4. Optimize to Minimize Loss**: Gradient Descent Optimizer

    **Optimize**
        Modify the weights and bias to improve the predictability of the model.

    .. code-block:: python
        :caption: Initialize the training step

        # each training step in gradient decent we want to minimize cross entropy
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # `0.5` is the learn rate

Conduct Training
~~~~~~~~~~~~~~~~

**1. Create Session and initialize global variables**

    .. code-block:: python

        # initialize the global variables
        init = tf.global_variables_initializer()

        # create an interactive session that can span multiple code blocks.  Don't
        # forget to explicity close the session with sess.close()
        sess = tf.Session()

        # perform the initialization which is only the initialization of all global variables
        sess.run(init)

**2. Training steps**

    .. code-block:: python

        # Perform 1000 training steps
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)    # get 100 random data points from the data. batch_xs = image,
                                                                # batch_ys = digit(0-9) class
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # do the optimization with this data

**3. Evaluate the model**

    .. code-block:: python

        # Evaluate how well the model did. Do this by comparing the digit with the highest probability in
        #    actual (y) and predicted (y_).
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


Deep Neural Network
-------------------

With the simple Neural Network above, the image is being represented as linear data.
This results in a loss of the information about the location of the pixel.

    **Convolution Layer**
        An added layer that looks at groups of pixels at a time

    **Pool Layer**
        Reduces the input to a smaller output

    **Fully Connected Layer**
        A layer consisting of neurons with all connections between its input and output

    **Over Fitting**
        A situation that occurs when the model is too well trained on the training data that it doesn't perform well on actual data.
        This can be resolved by setting a few of the weights and bias in the fully connected layer to 0.

Implementation
~~~~~~~~~~~~~~

**1. Prepared Data**: MNIST Data and reshaped as required

    .. code-block:: python
        :caption: As the simple neural network, define placeholders

        # Create input object which reads data from MNIST datasets.  Perform one-hot encoding to define the digit
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # Using Interactive session makes it the default sessions so we do not need to pass sess
        sess = tf.InteractiveSession()

        # Define placeholders for MNIST input data
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Note: now using an interactive session so that `sess` doesn't need to be repeatedly called.

    .. code-block:: python
        :caption: Convert linear data to a usable value cube for the convolution layer

        # change the MNIST input data from a list of values to a 28 pixel X 28 pixel X 1 grayscale value cube
        #    which the Convolution NN can use.
        x_image = tf.reshape(x, [-1,28,28,1], name="x_image")

        # `-1` is a flag to place a list of the other dimensions.
        # [batch, in_height, in_width, in_channels]

    .. code-block:: python
        :caption: Define helper functions for weight and bias initialization

        # Define helper functions to created weights and baises variables, and convolution, and pooling layers
        #   We are using RELU as our activation function.  These must be initialized to a small positive number
        #   and with some noise so you don't end up going to zero when comparing diffs
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # Weight shape: [filter_height, filter_width, in_channels, out_channels]
        #   channel is also referred to as features

    .. code-block:: python
        :caption: Define helper functions for Convolution and Pooling

        #   Convolution and Pooling - we do Convolution, and then pooling to control overfitting
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        # The stride parameter is how far and in which direction we shift as we compute new feature values
        # K size is the kernel size, which is the area we are pooling together
        #    [batch, in_height, in_width, in_channels] - maps to the input tensor

**2. Inference**: Matmul(x, Weight) + bias for entire NN

    .. code-block:: python
        :caption: Define first Convolution/Pool layer

        # 1st Convolution layer
        # 32 features for each 5X5 patch of the image
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        # Do convolution on images, add bias and push through RELU activation
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # take results and run through max_pool
        h_pool1 = max_pool_2x2(h_conv1)

        # Note, the result will be a 14x14 image

    **ReLU Activation Function**
        Bounds values from 0 to 1, where any negative values become zero.

    .. code-block:: python
        :caption: Define second Convolution/Pool layer

        # 2nd Convolution layer
        # Process the 32 features from Convolution layer 1, in 5 X 5 patch.  Return 64 features weights and biases
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        # Do convolution of the output of the 1st convolution layer.  Pool results
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Note, the result will be a 7x7 image

    .. code-block:: python
        :caption: Define the Fully Connected layer

        # Fully Connected Layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        # Weight Shape: [Input Size * Features/Channels, FC Neurons]

        #   Connect output of pooling layer 2 as input to full connected layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout some neurons to reduce overfitting
        keep_prob = tf.placeholder(tf.float32)  # get dropout probability as a training input.
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    .. code-block:: python
        :caption: Define Readout layer to convert all 1024 channels to the 10 digit output

        # Readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        # Define model
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

**3. Loss Measurement**: Cross Entropy

    .. code-block:: python
        :caption: Compare predicted digit ``y`` with actual digit ``y_`` then return the reduced mean

        # Loss measurement
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

        # Returns the mean of all the losses between the comparisons

**4. Optimize to Minimize Loss**: Adam Optimizer

    **Optimize**
        Modify the weights and bias to improve the predictability of the model.

    .. code-block:: python
        :caption: Initialize the training step

        # loss optimization
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Note: Using Adam instead of Gradient Descent

Conduct Training
~~~~~~~~~~~~~~~~

**1. Initialize global variables**

    .. code-block:: python

        # What is correct
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        # How accurate is it?
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize all of the variables
        sess.run(tf.global_variables_initializer())

        # Train the model
        import time

        #  define number of steps and how often we display progress
        num_steps = 3000
        display_every = 100

        # Start timer
        start_time = time.time()
        end_time = time.time()

**2. Training steps**

    .. code-block:: python

        for i in range(num_steps):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            # Periodic status display
            if i%display_every == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                end_time = time.time()
                print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))

        # Note: also feeding in keep_prob to randomly drop neurons during training to prevent over-fitting to the training data.

**3. Evaluate the model**

    .. code-block:: python

        # Display summary
        #     Time to train
        end_time = time.time()
        print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

        #     Accuracy on test data
        print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

        sess.close()

Methods Used
------------

This is a list of methods I have used when creating neural networks along with a quick description

`tf.placeholder(dtype, shape) <https://www.tensorflow.org/api_docs/python/tf/placeholder>`_
    Instantiates a tensor that will be fed into the network
`tf.Variable(shape) <https://www.tensorflow.org/versions/master/api_docs/python/tf/Variable>`_
    Instantiates a tensor that maintains state in the graph across calls to ``run()``
`tf.constant(value, shape) <https://www.tensorflow.org/versions/master/api_docs/python/tf/constant>`_
    Instantiates a constant tensor
`tf.matmul(tensor1, tensor2) <https://www.tensorflow.org/versions/master/api_docs/python/tf/matmul>`_
    Performs matrix multiplication
`tf.nn.softmax(tensor) <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/softmax>`_
    Performs SoftMax activation function on tensor
`tf.nn.softmax_cross_entropy_with_logits(labels, logits) <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_
    Calculates loss for each value using Cross Entropy
`tf.reduce_mean(tensor) <https://www.tensorflow.org/versions/master/api_docs/python/tf/reduce_mean>`_
    Calculates mean of elements across dimensions
`tf.tf.train.GradientDescentOptimizer(learn-rate).minimize(loss) <https://www.tensorflow.org/versions/master/api_docs/python/tf/train/GradientDescentOptimizer>`_
    Implements training step
`tf.global_variables_initializer() <https://www.tensorflow.org/versions/master/api_docs/python/tf/global_variables_initializer>`_
    Initializes global variables
`tf.Session() <https://www.tensorflow.org/versions/r1.0/api_docs/java/reference/org/tensorflow/Session>`_
    Initializes session, remember to close().
`tf.InteractiveSession() <https://www.tensorflow.org/versions/master/api_docs/python/tf/InteractiveSession>`_
    Initializes session without needing to call ``run()`` continuously.
`tf.reshape(input-tensor, shape, name) <https://www.tensorflow.org/versions/master/api_docs/python/tf/reshape>`_
    - Changes the shape of a tensor to another based on shape passed in.
    - Shape format: [batch, in_height, in_width, in_channels]
    - ``-1`` means that the size is calculated
`tf.truncated_normal(shape, stddev) <https://www.tensorflow.org/versions/master/api_docs/python/tf/truncated_normal>`_
    Returns tensor with random values from a normal distribution
`tf.nn.conv2d(input, filter, strides, padding) <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/conv2d>`_
    Computes a 2-D convolution given 4-D input and filter tensors
`tf.nn.max_pool(input, ksize, strides, padding) <>`_
    Computes a pool from input
`tf.nn.dropout(tensor, keep_prob) <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/dropout>`_
    Randomly removes weights and bias to reduce overfitting

