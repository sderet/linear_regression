import argparse
import math
import numpy
import matplotlib.pyplot as pyplot
import warnings

warnings.filterwarnings("error")

def normalize(data):
    transposed_data = numpy.transpose(data)

    normalized_data = []
    for line in transposed_data:
        line = (line - min(line)) / (max(line) - min(line))
        normalized_data.append(line)

    normalized_data = numpy.transpose(normalized_data)

    return normalized_data

def restore_scale(bias, slope, data):
    slope = (max(data[:, 1]) - min(data[:, 1])) * slope / (max(data[:, 0]) - min(data[:, 0]))
    bias = min(data[:, 1]) + ((max(data[:, 1]) - min(data[:, 1])) * bias) + slope * (1 - min(data[:, 0]))

    return bias, slope

def main(input, output, learning_rate, epochs, graph, verbose):
    try:
        data_content = numpy.loadtxt(input, dtype=float, delimiter=',', skiprows=1)
    except FileNotFoundError as e:
        print(f"{input} not found.")
        if verbose:
            print(f"Exiting...")
        exit()
    except ValueError as e:
        print(f"{input} is not properly formatted.")
        if verbose:
            print(f"{e}\nExiting...")
        exit()
    except UserWarning as e:
        print(f"{input} is not properly formatted. Is it one line or less?")
        if verbose:
            print(f"{e}\nExiting...")
        exit()

    data_size = len(data_content)

    # Necessary to prevent data becoming too big to do math on
    normalized_data = normalize(numpy.array(data_content))

    bias = 0
    slope = 0

    for _ in range(0, epochs):
        tmp_bias = 0
        tmp_slope = 0

        mse = 0

        for values in normalized_data:
            current_predict = bias + (slope * values[0])

            mse += (current_predict - values[1]) ** 2

            tmp_bias -= (current_predict - values[1])
            tmp_slope -= (current_predict - values[1]) * values[0]

        mse = mse / len(normalized_data)

        bias += learning_rate * (tmp_bias / data_size)
        slope += learning_rate * (tmp_slope / data_size)

    #De-normalize bias and slope
    bias, slope = restore_scale(bias, slope, data_content)

    print(f"Bias: {bias}\nSlope: {slope}")

    numpy.savetxt(output, numpy.array([bias, slope]).T)

    
    #plot
    pyplot.plot(data_content[:, 0], data_content[:, 1], "o")
    pyplot.axline([0, bias], slope=slope, color="r")
    pyplot.title("Linear regression")
    pyplot.xlabel("km")
    pyplot.ylabel("price")
    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="datasets/data.csv", help="the file containing data to perform a linear regression on")
    parser.add_argument("-o", "--output", default="weight.lreg", help="the destination file to create containing the weights and bias")
    parser.add_argument("-r", "--learning-rate", type=float, default=0.05, help="learning rate in the linear regression")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="the number of iterations to perform gradient descent for")
    parser.add_argument("-g", "--graph", help="display a graph with the data points and the resulting function", action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    main(args.input, args.output, args.learning_rate, args.epochs, args.graph, args.verbose)