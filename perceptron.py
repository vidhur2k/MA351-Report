"""
This is an implementation of a simple perceptron algorithm to
classify linearly seperable data.

Authors: Harjas Monga and Vidhur Kumar
"""


# Imports
from __future__ import print_function
import sys
from matplotlib import pyplot as plt
import numpy as np


# Returns 1 if the weighted input sum is greater than a threshold.
def predict(inputs, weights):
    activation_threshold = 0.0

    # Obtaining the weighted sum of the inputs.
    for i, w in zip(inputs, weights):
        activation_threshold += w * i

    return 1.0 if activation_threshold >= 0.0 else 0.0


def plot(matrix, weights=None, title="Prediction Matrix"):

    # If the input vector is 1D
    if len(matrix[0]) == 3:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Input")
        ax.set_ylabel("Classifications")

        y_min = 0.0
        y_max = 1.1
        x_min = 0.0
        x_max = 1.1
        y_res = 0.001
        x_res = 0.001

        if weights != None:

            ys = np.arange(y_min, y_max, y_res)
            xs = np.arange(x_min, x_max, x_res)
            zs = []

            for currentY in np.arange(y_min, y_max, y_res):
                for currentX in np.arange(x_min, x_max, x_res):
                    zs.append(predict([1.0, currentX], weights))
            xs, ys = np.meshgrid(xs, ys)
            zs = np.array(zs)
            zs = zs.reshape(xs.shape)
            cp = plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('b','r'), alpha=0.1)

        c1_data = [[], []]
        c0_data = [[], []]

        for i in range(len(matrix)):
            cur_i1 = matrix[i][1]
            cur_y = matrix[i][-1]

            # If the output class is 1
            if cur_y == 1:
                c1_data[0].append(cur_i1)
                c1_data[1].append(1.0)

            # If the output class is 0
            else:
                c0_data[0].append(cur_i1)
                c0_data[1].append(0.0)

        plt.xticks(np.arange(x_min, x_max, 0.1))    # Plot the ticks on the x-axis
        plt.yticks(np.arange(y_min, y_max, 0.1))    # Plot the ticks on the y-axis

        plt.xlim(0, 1.05)
        plt.ylim(-0.05, 1.05)

        c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label='Class 0')
        c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label='Class 1')

        plt.legend(fontsize=10, loc=1)
        plt.show()
        return

    # Input is 2D
    if len(matrix[0]) == 4:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        if weights != None:
            map_min = 0.0
            map_max = 1.1
            x_res = 0.001
            y_res = 0.001
            ys = np.arange(map_min, map_max, y_res)
            xs = np.arange(map_min, map_max, x_res)
            zs = []
            for currentY in np.arange(map_min, map_max, y_res):
                for currentX in np.arange(map_min, map_max, x_res):
                    zs.append(predict([1.0, currentX, currentY], weights))
            xs, ys = np.meshgrid(xs, ys)
            zs = np.array(zs)
            zs = zs.reshape(xs.shape)
            cp = plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('b', 'r'), alpha=0.1)

        c1_data = [[], []]
        c0_data = [[], []]
        for i in range(len(matrix)):
            cur_i1 = matrix[i][1]
            cur_i2 = matrix[i][2]
            cur_y = matrix[i][-1]
            if cur_y == 1:
                c1_data[0].append(cur_i1)
                c1_data[1].append(cur_i2)
            else:
                c0_data[0].append(cur_i1)
                c0_data[1].append(cur_i2)

        plt.xticks(np.arange(0.0, 1.1, 0.1))
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)

        c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label='Class 0')
        c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label='Class 1')

        plt.legend(fontsize=10, loc=1)
        plt.show()
        return


    print("Invalid vector dimension provided.")

# Returns the percentage accuracy of the classifier.
def accuracy(matrix, weights):
    num_correct = 0.0
    preds = []

    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1], weights)
        preds.append(pred)

        if pred == matrix[i][-1]:
            num_correct += 1.0

    print("Predictions:", preds)

    return num_correct / float(len(matrix))

# Backpropogation algorithm
def train_weights(matrix, weights, n_epoch=10, l_rate=1.00, do_plot=False, stop_early=True, verbose=True):

    for epoch in range(n_epoch):
        current_accuracy = accuracy(matrix, weights)
        print("\nEpoch %d \nWeights: " %epoch, weights)
        print("Accuracy: ", current_accuracy)

        # If the maximum accuracy is reached
        if current_accuracy == 1.0 and stop_early:
            break

        if do_plot:
            plot(matrix, weights, title="Epoch %d" %epoch)

        for i in range(len(matrix)):

            # Current prediction
            prediction = predict(matrix[i][:-1], weights)

            # Error in prediction
            error = matrix[i][-1] - prediction

            if verbose:
                sys.stdout.write("Training on data at index %d...\n"%(i))

            # Updating the weights
            for j in range(len(weights)):
                if verbose:
                    sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j, weights[j]))


                weights[j] += (l_rate * error * matrix[i][j])

                if verbose:
                    sys.stdout.write("%0.5f\n"%(weights[j]))

    plot(matrix, weights, title="Final epoch")
    return weights

# Testing the two input plotting algorithm.
def main():

    n_epoch = 10
    l_rate = 1.0
    do_plot = False
    stop_early = True

    """
    Let x be a data vector. The format of the data is the following:

        x = [b, i1, i2, y] where b is the bias, i1 and i2 are the
        point coordinates ((i1, i2 => (x, y)) essentially, and y is the output class.
    """
            # b    # i1    # i2    # y
    matrix = [[1.0, 0.08, 0.72, 1.0],
              [1.0, 0.10, 1.00, 0.0],
              [1.0, 0.26, 0.58, 1.0],
              [1.0, 0.35, 0.95, 0.0],
              [1.0, 0.45, 0.15, 1.0],
              [1.0, 0.60, 0.30, 1.0],
              [1.0, 0.70, 0.65, 0.0],
              [1.0, 0.92, 0.45, 0.0]]
                # i1    # i2    # y
    weights = [0.20, 1.00, -1.00]

    train_weights(matrix, weights=weights, n_epoch=n_epoch, l_rate=l_rate, do_plot=do_plot,
                  stop_early=stop_early)


if __name__ == '__main__':
    main()
