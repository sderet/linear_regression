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

def calculate_mse(bias, slope, values, data_content):
    bias, slope = restore_scale(bias, slope, data_content)
    current_predict = bias + (slope * values[0])
    return (current_predict - values[1]) ** 2

def main(input, output, learning_rate, epochs, minimum_improvement, graph, verbose):
    try:
        if epochs < 1:
            raise ValueError(f"total epochs cannot be less than 1; {epochs} was provided")

        if learning_rate > 1:
            raise ValueError(f"learning rate cannot exceed 1.0; {learning_rate} was provided")
        
        data_content = numpy.atleast_2d(numpy.loadtxt(input, dtype=float, delimiter=',', skiprows=1))
        data_size = len(data_content)
        if (data_size <= 1):
            raise ValueError(f"at least 2 datapoints are expected; {data_size} was provided")
    except FileNotFoundError as e:
        print(f"Error: {e}\nExiting...")
        exit()
    except ValueError as e:
        print(f"Error: {e}\nExiting...")
        exit()
    except UserWarning as e:
        print(f"Error: {e}\nExiting...")
        exit()

    # Necessary to prevent data becoming too big to do math on
    normalized_data = normalize(numpy.array(data_content))

    bias = 0
    slope = 0
    mse = 0
    if graph > 1:
        blist = []
        slist = []

    for current_epoch in range(0, epochs):
        tmp_bias = 0
        tmp_slope = 0

        prev_mse = mse
        mse = 0

        for i, values in enumerate(normalized_data):
            mse += calculate_mse(bias, slope, data_content[i], data_content)

            current_predict = bias + (slope * values[0])

            tmp_bias -= (current_predict - values[1])
            tmp_slope -= (current_predict - values[1]) * values[0]

        mse = mse / len(normalized_data)
        improvement = prev_mse - mse

        bias += learning_rate * (tmp_bias / data_size)
        slope += learning_rate * (tmp_slope / data_size)


        tmp_bias, tmp_slope = restore_scale(bias, slope, data_content)

        if graph > 1:
            blist.append(tmp_bias)
            slist.append(tmp_slope)

        if verbose:
            print(f"  Epoch #{current_epoch + 1}\nBias = {tmp_bias}, slope = {tmp_slope}, MSE = {mse}")
            if current_epoch > 0:
                print(f"MSE improved by {improvement}")

        if current_epoch > 0 and improvement >= 0 and improvement < minimum_improvement:
            print(f"Reached minimum improvement; MSE improvement on epoch #{current_epoch + 1} is {improvement}")
            break

    # De-normalize bias and slope
    bias, slope = restore_scale(bias, slope, data_content)

    print(f"  Final values:\nBias: {bias}\nSlope: {slope}\nMean squared error: {mse}")

    numpy.savetxt(output, numpy.array([bias, slope]).T)

    if graph:
        pyplot.plot(data_content[:, 0], data_content[:, 1], "o")
        ax = pyplot.axline([0, bias], slope=slope, color="r")
        pyplot.title("Linear regression")
        pyplot.xlabel("km")
        pyplot.ylabel("price")

        if graph > 1:
            pyplot.ion()

            for i in range(len(slist)):
                if not pyplot.get_fignums():
                    exit()
                pyplot.title(f"Linear regression (epoch #{i + 1})")
                ax.set_xy1([0, blist[i]])
                ax.set_slope(slist[i])
                # Take 5 seconds no matter how many epochs there are
                pyplot.pause(5 / current_epoch)

            pyplot.ioff()
            pyplot.show()
        else:
            pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", default="datasets/data.csv", help="the file containing data to perform a linear regression on")
    parser.add_argument("-o", "--output", default="weight.lreg", help="the destination file to create containing the weights and bias")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.2, help="learning rate in the linear regression")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="the maximum number of iterations to perform gradient descent for")
    parser.add_argument("-m", "--minimum-improvement", type=float, default=0, help="if set, stops running once mse improvement between epochs is less than the given value")
    parser.add_argument("-g", "--graph", help="display a graph with the data points and the resulting function. If used more than once, will display the function at every epoch in sequence", action='count', default=0)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    main(args.input, args.output, args.learning_rate, args.epochs, args.minimum_improvement, args.graph, args.verbose)