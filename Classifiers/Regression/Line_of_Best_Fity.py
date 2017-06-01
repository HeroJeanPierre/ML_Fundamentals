from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# This will be used to create random data sets
def create_dataset(size, varience, step=2, correlation=False):
    val = 0
    ys = []

    # Assign random values to the ys list
    for i in range(size):
        y = val + random.randrange(-varience, varience)
        ys.append(y)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         (mean(xs) * mean(xs) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


# This is meant to tell us how good of a fit the line
# We made is
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr / squared_error_y_mean)

# Create our data set
xs, ys = create_dataset(1000, 100, 2, correlation='pos')

# Make the line of best fit
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = []

for x in xs:
    regression_line.append((m * x) + b)

# This is how we know how good of a fit the line actually is
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# Show the graph
plt.scatter(xs, ys, s=10)
plt.plot(xs, regression_line)
plt.show()
