import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as scipyRotate

sys.path.append(os.path.abspath('.\\'))
from module import neuralnet as nn


if __name__ == '__main__':

    INPUT_NODE = 784
    HIDDEN_NODE = 128
    OUTPUT_NODE = 10

    LEARNING_RATE = 0.01

    NEURAL_NETWORK = nn.NeuralNetwork(INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE)

    with open("src/mnist_train.csv") as f:
        TRAIN_DATA = f.readlines()

    with open("src/mnist_test.csv") as f:
        TEST_DATA = f.readlines()


    EPOCH = 10
    ACCURACY = []
    for e in range(EPOCH):
        for record in TRAIN_DATA:

            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # image rotation
            inputs_plus10 = scipyRotate(inputs.reshape(28, 28), 10, cval=0.01, reshape=False).reshape(-1,)
            inputs_minus10 = scipyRotate(inputs.reshape(28, 28), -10, cval=0.01, reshape=False).reshape(-1,)

            #  labeling
            correct_label = int(all_values[0])
            targets = np.zeros(OUTPUT_NODE) + 0.01
            targets[correct_label] = 0.99

            # train
            NEURAL_NETWORK.task(inputs, sel=1, target_list=targets, learning_rate=LEARNING_RATE)
            NEURAL_NETWORK.task(inputs_plus10, sel=1, target_list=targets, learning_rate=LEARNING_RATE)
            NEURAL_NETWORK.task(inputs_minus10, sel=1, target_list=targets, learning_rate=LEARNING_RATE)

        TEST_RESULT = np.array([])

        for record in TEST_DATA:
            all_values = record.split(',')
            correct_label = int(all_values[0])

            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = NEURAL_NETWORK.task(inputs, sel=0)
            label = np.argmax(outputs)

            if label == correct_label:
                TEST_RESULT = np.append(TEST_RESULT, 1)
            else:
                TEST_RESULT = np.append(TEST_RESULT, 0)

        ACCURACY.append(TEST_RESULT.sum() / TEST_RESULT.size)



    print("Accuracy : ", ACCURACY)


    print("weights(input-hidden) : ", NEURAL_NETWORK.weight_ih.shape, "weights(hidden-output) : ", NEURAL_NETWORK.weight_ho.shape)


    with open('src/weights2.csv', 'w') as f:

        np.savetxt(f, NEURAL_NETWORK.weight_ih, fmt='%.3f', delimiter=',')
        np.savetxt(f, NEURAL_NETWORK.weight_ho, fmt='%.3f', delimiter=',')

    plt.plot(range(EPOCH), ACCURACY, 'ro', range(EPOCH), ACCURACY, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('3-layers Neural Network(hidden node = 128)')
    plt.grid(axis='y')
    plt.show()
