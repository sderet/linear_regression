import argparse
import numpy

def main(x, input, verbose):

    try:
        parameters = numpy.genfromtxt(input, dtype=float)
        if len(parameters) != 2:
            raise ValueError(f"expected 2 parameters, got {len(parameters)}")
    except FileNotFoundError:
        print(f"Input file not found. Using 0 as weight and bias.")
        parameters = [0, 0]
    except ValueError as e:
        print(f"Input file {input} is incorrectly formatted.")
        if verbose:
            print(f"Error: {e}\nExiting...")
        exit()

    prediction = parameters[0] + (parameters[1] * x)

    print(f"The expected value for {x} is {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("value", type=int, help="the value at which a the intercept will be predicted")
    parser.add_argument("-i", "--input", default="weight.lreg", help="the input file containing the weights and bias")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    main(args.value, args.input, args.verbose)