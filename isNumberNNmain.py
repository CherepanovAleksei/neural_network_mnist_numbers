import numpy as np
from mnist_data_reader import MNIST


def sig(x, derivative=False):
    if derivative is True:
        return sig(x) * (1 - sig(x))
    return 1 / (1 + np.exp(-x))


# load training and testing data
data = MNIST('./input_data')
train_images = data.load_training()
test_images = data.load_testing()

# init right results for training
out_true = np.ones((28, 28), dtype=np.float)
out_false = np.zeros((28, 28), dtype=np.float)

# random initial weights (2 layer)
weight0 = 2 * np.random.random((28, 28)) - 1
weight1 = 2 * np.random.random((28, 28)) - 1

# training (backpropagation)
for count in range(60000):
    # true answer
    img_in = np.asarray(train_images[count])
    img_in.shape = (28, 28)
    layer0 = img_in
    layer1 = sig(np.dot(layer0, weight0))
    layer2 = sig(np.dot(layer1, weight1))

    l2_error = out_true - layer2
    l2_delta = l2_error * sig(layer2, derivative=True)

    l1_error = l2_delta.dot(weight1.T)

    l1_delta = l1_error * sig(layer1, derivative=True)

    weight1 += layer1.T.dot(l2_delta)
    weight0 += layer0.T.dot(l1_delta)

    # false
    img_in = np.random.randint(0, 255, (28, 28))
    layer0 = img_in
    layer1 = sig(np.dot(layer0, weight0))
    layer2 = sig(np.dot(layer1, weight1))

    l2_error = out_false - layer2
    l2_delta = l2_error * sig(layer2, derivative=True)

    l1_error = l2_delta.dot(weight1.T)

    l1_delta = l1_error * sig(layer1, derivative=True)

    weight1 += layer1.T.dot(l2_delta)
    weight0 += layer0.T.dot(l1_delta)

# testing
with open('output/right_output.txt',  'w') as right_output:
    for count in range(10000):
        img_in = np.asarray(test_images[count])
        img_in.shape = (28, 28)
        layer0 = img_in
        layer1 = sig(np.dot(layer0, weight0))
        layer2 = sig(np.dot(layer1, weight1))
        right_output.write("%.10f" % layer2.mean())
        right_output.write(data.display(test_images[count]) + '\n\n')

with open('output/wrong_output.txt', 'w') as wrong_output:
    for count in range(10000):
        img_in = np.random.randint(0, 255, (28, 28))
        layer0 = img_in
        layer1 = sig(np.dot(layer0, weight0))
        layer2 = sig(np.dot(layer1, weight1))
        wrong_output.write("%.10f" % layer2.mean())
        img_in.shape = 784
        wrong_output.write(data.display(img_in) + '\n\n')
