import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd()))

from module import neuralnet as nn



if __name__ == '__main__':
    INPUT_NODE = 784
    HIDDEN_NODE = 128
    OUTPUT_NODE = 10


    with open('src/weights.csv', 'r') as f:
        WEIGHTS = [line.rstrip("\n") for line in f.readlines()]

    WEIGHT_IH = [element.split(',') for element in WEIGHTS[0 : HIDDEN_NODE]]
    WEIGHT_HO = [element.split(',') for element in WEIGHTS[HIDDEN_NODE : HIDDEN_NODE+OUTPUT_NODE]]
    WEIGHT_IH = np.asfarray(WEIGHT_IH)
    WEIGHT_HO = np.asfarray(WEIGHT_HO)

    NEURAL_NET = nn.NeuralNetwork(INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE, WEIGHT_IH, WEIGHT_HO)

    with open("src/mnist_test.csv") as f:
        TEST_DATA = f.readlines()

    TEST_RESULT = np.array([])
    for record in TEST_DATA:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = NEURAL_NET.task(inputs, sel=0)
        label = np.argmax(outputs)

        if label == correct_label:
            TEST_RESULT = np.append(TEST_RESULT, 1)
        else:
            TEST_RESULT = np.append(TEST_RESULT, 0)

    print("Accuracy : ", TEST_RESULT.sum() / TEST_RESULT.size)

    #  back query

    LABEL_IDX = 0
    LABEL_ARRAY = np.zeros(OUTPUT_NODE) + 0.01
    LABEL_ARRAY[LABEL_IDX] = 0.99

    BQ_RESULT = NEURAL_NET.backquery(LABEL_ARRAY)


    plt.imshow(BQ_RESULT.reshape(28, 28), cmap='Greys', interpolation=None)
    plt.show()
