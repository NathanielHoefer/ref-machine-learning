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