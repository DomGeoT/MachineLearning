import numpy
import matplotlib.pyplot as plt
import math

global coefficients
global learningRate
X = [lambda x : 1, lambda x : x, lambda x : x ** 2, lambda x : x ** 3]
coefficients =  [0] * len(X)
learningRate = 0.000005

def f(variable):
    global coefficients
    sum = 0
    for i in range(0, len(X)):
        sum += X[i](variable) * coefficients[i]
    return sum

def nextCoefficients():
    global coefficients
    global learningRate

    for i in range(0, len(coefficients)):
        gradient = 0
        for j in range(0, len(data[0])):
            gradient += (data[1][j] - f(data[0][j]))
        gradient = gradient * -2 * data[0][i]
        coefficients[i] = coefficients[i] - learningRate * gradient

def getData(filename, separator):
    '''
    columns 0 - n-3: Predictors
    column  n-2:     Outcome
    column n-1:      Train/test indicator

    :return:
    '''
    dataFile = open(filename, 'r')
    predictors = []
    outcome = []
    for line in dataFile:
        splitData = line.strip().split(separator)
        if len(line) == 0:
            continue
        predictors.append(splitData[0])
        outcome.append(splitData[1])

    for i in range(0, len(predictors)):
        predictors[i] = float(predictors[i])
        outcome[i] = float(outcome[i])

    return predictors, outcome


def squaredLoss(data):
    global coefficients
    loss = 0
    for i in range(0, len(data[0])):
        loss += math.fabs(data[1][i] - f(data[0][i]))
    return loss


def plotData(x, y):
    plt.plot(x, y, 'ro')

    p = numpy.linspace(-50, 50, 1000)

    plt.plot(p, f(p))
    plt.show()

data = getData('lineData.csv', ',')
count = 1

prevLoss = 1000000000000

while True:
    print(squaredLoss(data))
    nextCoefficients()

    if (prevLoss - squaredLoss(data))**2 < 10:
        break
    prevLoss = squaredLoss(data)

    print(count)
    count += 1

print(squaredLoss(data))
print(coefficients)
plotData(data[0], data[1])