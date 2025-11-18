# Linear regression

A simple Python program to compute and visualize linear regressions, made for the 42 cursus.

## Usage

### Installing dependencies

Run `pip install -r requirements.txt`. If not planning on using the graph view, you may delete `matplotlib` from the requirements beforehand.

### Basic usage

>Use `python train.py -h` and `python predict.py -h` for a full list of available options.

Run `python train.py` to train your linear regression. Then, run `python predict.py [X]` to get a prediction for X.

The default file containing the bias and slope from the linear regression is `weight.lreg`. This can be modified with the `-o [output]` flag.

### Graphical view

By adding the `-g` flag, a graph will be displayed with the datapoints as well as the final function from your linear regression.

By using `-gg`, an animation will be displayed instead, showing the evolution of the function through the epochs.