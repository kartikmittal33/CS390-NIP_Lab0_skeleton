import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"

# Neurons per layer
NEURONS = 512

np.set_printoptions(threshold=np.inf)


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.05):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sig = self.__sigmoid(x)
        der = sig * (1 - sig)
        return der  # TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
    def train(self, xVals, yVals, epochs=100, minibatches=True, mbs=100):
        num_batches = xVals.shape[0] / mbs
        xValBatches = np.split(xVals, num_batches)
        yValBatches = np.split(yVals, num_batches)
        for i in range(epochs):
            for j in range(num_batches):
                layer1, layer2 = self.__forward(xValBatches[j])
                L2e = (layer2 - yValBatches[j])

                sig_der_layer2 = self.__sigmoidDerivative(layer2)
                L2d = L2e * sig_der_layer2

                L1e = np.dot(L2d, self.W2.T)

                sig_der_layer1 = self.__sigmoidDerivative(layer1)
                L1d = L1e * sig_der_layer1

                L1a = (np.dot(xValBatches[j].T, L1d)) * self.lr
                L2a = (np.dot(layer1.T, L2d)) * self.lr

                self.W1 -= L1a
                self.W2 -= L2a

        return self

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        ans = []
        for entry in layer2:
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            index = entry.argmax()
            pred[index] = 1
            ans.append(pred)

        return np.array(ans)


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = (
        (raw[0][0] / 255.0, raw[0][1]),
        (raw[1][0] / 255.0, raw[1][1]))  # TODO: Add range reduction here (0-255 ==> 0.0-1.0).

    xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])
    xTest = xTest.reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        # TODO: Write code to build and train your custon neural net.
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, NEURONS)
        return model.train(xTrain, yTrain)
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        # TODO: Write code to build and train your keras neural net.
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
             tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=50)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        # TODO: Write code to run your custon neural net.
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        # TODO: Write code to run your keras neural net.
        preds = model.predict(data)
        ans = []
        for entry in preds:
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            index = entry.argmax()
            pred[index] = 1
            ans.append(pred)
        return np.array(ans)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / (preds.shape[0] * 1.0)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
