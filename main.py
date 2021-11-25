# Jerry Xu
# CS 4375 Fall 2021 Homework 4 Part 1 (Perceptron)
import math
import sys


def dot(v1, v2):
    """
    returns the dot product of vectors v1 and v2
    """
    assert len(v1) == len(v2)
    return sum(a * b for a, b in zip(v1, v2))


def add(v1, v2):
    """
    returns the sum of vectors v1 and v2
    """
    assert len(v1) == len(v2)
    return list(a + b for a, b in zip(v1, v2))


def mul(v, s):
    """
    returns the product of a vector v and a scalar s
    """
    return list(map(lambda x: x * s, v))


def sigma(x):
    """
    numerically stable sigmoid function from https://stackoverflow.com/a/25164452
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def sigma_prime(x):
    """
    returns sigma'(x)
    """
    return sigma(x) * (1 - sigma(x))


def update_weights(weights, example, learning_rate):
    """
    updates the weights of the perceptron
    """
    x, y = example[:-1], example[-1]
    w_dot_x = dot(weights[-1], x)
    delta = y - sigma(w_dot_x)

    return add(weights[-1], mul(x, learning_rate * delta * sigma_prime(w_dot_x)))


def get_output(weights, example):
    """
    returns the output of the perceptron when given an example
    """
    return sigma(dot(weights[-1], example[:-1]))


def print_outputs(weights, train_dataset, learning_rate, num_iterations, attributes):
    """
    prints the results of training to screen
    """
    for i in range(num_iterations):
        curr_example = train_dataset[i % len(train_dataset)]
        new_weights = update_weights(weights, curr_example, learning_rate)
        weights.append(new_weights)
        output = get_output(weights, curr_example)

        line = "After iteration {}: ".format(i + 1)
        for j in range(len(attributes)):
            line += "w({}) = {:.4f}, ".format(attributes[j], new_weights[j])
        line += "output = {:.4f}".format(output)

        print(line)


def predict(weights, example):
    """
    returns the prediction of the perceptron when given an example
    """
    return 1 if get_output(weights, example) >= 0.5 else 0


def accuracy(weights, dataset):
    """
    returns the accuracy of the perceptron when tested on a dataset
    """
    correct = 0

    for example in dataset:
        if predict(weights, example) == example[-1]:
            correct += 1

    return correct / len(dataset)


def main():
    # command line args
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    learning_rate = float(sys.argv[3])
    num_iterations = int(sys.argv[4])

    # read the training and test sets from their respective files
    with open(train_file_name) as f:
        # some python magic to skip over all empty lines
        train_dataset = [line.split() for line in f if line.strip()]

    attributes = train_dataset.pop(0)

    # more python magic to get the dataset into a usable format
    # now the dataset is a list of lists in the following form: [1, 1, 0, 0, 1]
    # where the last element represents the class
    # and every other element represents an attribute value
    train_dataset = [list(map(int, example)) for example in train_dataset]

    with open(test_file_name) as f:
        test_dataset = [line.split() for line in f if line.strip()]

    test_dataset.pop(0)
    test_dataset = [list(map(int, example)) for example in test_dataset]

    # initialize the weights to 0
    weights = [[0] * (len(attributes) - 1)]

    # train the perceptron and print the results of training to screen
    print_outputs(weights, train_dataset, learning_rate, num_iterations, attributes[:-1])

    # test the trained perceptron on the training set
    train_accuracy = accuracy(weights, train_dataset)
    print("\nAccuracy on training set ({} instances): {:.1f}%".format(len(train_dataset), 100 * train_accuracy))

    # test the trained perceptron on the test set
    test_accuracy = accuracy(weights, test_dataset)
    print("\nAccuracy on test set ({} instances): {:.1f}%".format(len(test_dataset), 100 * test_accuracy))


if __name__ == "__main__":
    main()
