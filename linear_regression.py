'''
Linear regression is a linear approach for modeling the relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X.

Author: Kushal Luitel
Project: https://github.com/kushal12345/Linear-regression
'''
#importing necessary library
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Initialize Rnd with random number
rnd = np.random
#Parameters required
learning_rate = 0.01 #The learning rate is how quickly a network abandons old beliefs for new ones.
training_epochs = 1000 #An epoch is one complete presentation of the data set to be learned to a learning machine.
display_step = 50 #number of steps

#Training data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])

train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                     2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0] #The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
                       #In short this line is just counting the number of Input data

#tf Graph input
X = tf.placeholder("float") #A placeholder is simply a variable that we will assign data to at a later date.
Y = tf.placeholder("float")
#Initialize weights and bias


#Weight - Weight is the strength of the connection. If I increase the input then how much influence does it have on the output.
#         Weights near zero mean changing this input will not change the output. Many algorithms will automatically set those weights to zero in order to simplify the network.
#Bias   - as means how far off our predictions are from real values. Generally parametric algorithms have a high bias making them fast to learn and easier to understand but generally less flexible.
#          In turn they are have lower predictive performance on complex problems that fail to meet the simplifying assumptions of the algorithms bias.

            #Low Bias: Suggests more assumptions about the form of the target function.
#High-Bias: Suggests less assumptions about the form of the target function.
W = tf.Variable(rnd.randn(), name="Weight") # assigning random inital value to W and naming it.
b = tf.Variable(rnd.randn(), name="bias")

#Construct a linear model
pred = tf.add(tf.multiply(X,W), b) #(X*W+b) normal linear model

#mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

#gradien descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initialize the global variable with their default value
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:

    sess.run(init) # running the initializer

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:",'%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=",sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost,feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=",sess.run(b), '\n')

#Graphics display_step
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

# Testing example, as requested (Issue #2)
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
