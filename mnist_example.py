"""
MNIST dataset을 학습하는 예제
"""
import numpy as np
import os
import zipfile

from dl_numpy.layer.activation import softmax, gelu
from dl_numpy.layer.loss import cross_entropy
from dl_numpy.layer import Linear


if not os.path.isfile('./example_data/mnist/x_train.npy'):
    print('Extracting MNIST dataset...')
    with zipfile.ZipFile('./example_data/mnist/mnist.zip') as f:
        f.extractall('./example_data/mnist')

x_train = np.load('./example_data/mnist/x_train.npy')
y_train = np.load('./example_data/mnist/y_train.npy')
x_test = np.load('./example_data/mnist/x_test.npy')
y_test = np.load('./example_data/mnist/y_test.npy')


x_train = x_train.reshape((-1, 28*28)) / 255
x_test = x_test.reshape((-1, 28*28)) / 255
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


m1 = Linear(784, 128, activation=gelu)
m2 = Linear(128, 10, activation=softmax)

lr = 0.01
batch_size = 256
for epoch in range(100):
    for i in range(0, len(x_train), batch_size):
        # forward
        out = m1(x_train[i:i+batch_size])
        out = m2(out)

        # loss
        loss = cross_entropy(y_train[i:i+batch_size], out)

        # backward
        grad_dict = loss.backward()

        # apply
        m1.apply_grad(grad_dict, lr)
        m2.apply_grad(grad_dict, lr)

        print('loss:', loss)

