# most of this code is from:
# https://nbviewer.jupyter.org/github/jennselby/MachineLearningCourseNotes/blob/master/assets/ipynb/LinearRegression.ipynb

import matplotlib

import numpy.random
from sklearn import linear_model
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# setting x variable limits - I changed the limits

MIN_X = -30
MAX_X = 30
NUM_INPUTS = 100

# randomly picks numbers for x

x1 = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 1))

print(x1)

# displays the "fake x" data in a spreadsheet-like format

data = pd.DataFrame(data = x1, columns = ['x'])
data.head()

# follows the equation "y = 0.3x + 1"

data['y'] = 0.3 * data['x'] + 1

# plots the data in a 2D plane

data.plot.scatter(x='x', y='y')

# creating noise

noise = numpy.random.normal(size=NUM_INPUTS)

# creating the y variable with noise

data['y'] = data['y'] + noise

# plotting the realistic data

data.plot.scatter(x='x', y='y')

# training the model to find the best fit line

# creating the empty linear model

model1 = linear_model.LinearRegression()
x = data['x'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)

# fitting the data to be used for the line

model1.fit(X=x, y=y)

# displays the results of the training

def print_model_fit(model):
    # Print out the parameters for the best fit line
    print('Intercept: {i}  Coefficients: {c}'.format(i=model.intercept_, c=model.coef_))

# executing the function

print_model_fit(model1)

# print out guesses for x values

new_x_values = [ [-1.23], [0.66], [1.98]]

predictions = model1.predict(new_x_values)

print(predictions)

# using string formatting, print out the data 

for datapoint, prediction in zip(new_x_values, predictions):
    print('Model prediction for {}: {}'.format(datapoint[0], prediction))

# plotting the line of best fit

def plot_best_fit_line(model, x, y):
    # formatting the graph
    fig = matplotlib.pyplot.figure(1)
    fig.suptitle('Data and Best-Fit Line')
    matplotlib.pyplot.xlabel('x values')
    matplotlib.pyplot.ylabel('y values')
    
    # place points on the graph
    matplotlib.pyplot.scatter(x, y)
    
    #plot the line of best fit
    
    X = numpy.linspace(MIN_X, MAX_X) # generates all the possible values of x
    Y = model.predict(list(zip(X)))
    matplotlib.pyplot.plot(X, Y)

# plotting the line
plot_best_fit_line(model1, x, y)


