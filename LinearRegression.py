import numpy
import matplotlib.pyplot as plt
import random
import copy

X = [lambda x : 1, lambda x : x ** 3 ]

def f(variable, coefficients):
    sum = 0
    for i in range(0, len(X)):
        sum += X[i](variable) * coefficients[i]
    return sum


def nextCoefficients(data, coefficients, lossFunc, lam):
    learningRate = 0.0005

    for i in range(0, len(coefficients)):
        gradient = 0
        for j in range(0, len(data[0])):
            gradient += lossFunc(data[1][j] - f(data[0][j], coefficients), coefficients, lam)
        gradient = gradient * -1 * data[0][i]
        coefficients[i] = coefficients[i] - learningRate * gradient

def stndLoss(lossTerm, *args):
    return lossTerm

def lassoLoss(lossTerm, coefficients, lam):
    return lam * lossTerm + (1 - lam) * sum(coefficients)


def ridgeLoss(lossTerm, coefficients, lam):
    return lam * lossTerm + (1 - lam) * sum(list(numpy.power(coefficients, 2)))

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


def squaredLoss(data, coefficients):
    loss = 0
    for i in range(0, len(data[0])):
        loss += numpy.float_power(data[1][i] - f(data[0][i], coefficients), 2)
    return loss

def plotData(x, y, coefficients):
    plt.plot(x, y, 'ro')

    p = numpy.linspace(-50, 50, 1000)

    plt.plot(p, f(p, coefficients))
    plt.show()


def randomizeCoefficients(length, deviation, centre = 0):
    coefficients = [0] * length

    for i in range(len(coefficients)):
        coefficients[i] = centre + random.randint(-deviation * 100, deviation * 100)/100

    return coefficients


def runLinReg(data, coefficients, lossFunc=stndLoss, lam=0.5):
    print()
    print("initial coefficients", coefficients, "initial loss", squaredLoss(data, coefficients))
    count = 1
    prevLoss = 1000000000000

    while True:
        nextCoefficients(data, coefficients, lossFunc, lam)

        if numpy.power(prevLoss - squaredLoss(data, coefficients),2) < 0.0000001 or count > 5000:
            prevLoss = squaredLoss(data, coefficients)
            break
        prevLoss = squaredLoss(data, coefficients)

        count += 1


    print("iterations performed", count, "loss", prevLoss, "coefficients", coefficients)

    return prevLoss, coefficients


fileData = getData('squareData.csv', ',')

minLoss = 10000000
bestCoefficients = randomizeCoefficients(len(X), 1)

for i in range(10):
    l, c = runLinReg(fileData, randomizeCoefficients(len(X), 2, 2), lassoLoss, 0.5)
    if l < minLoss:
        minLoss = l
        bestCoefficients = copy.deepcopy(c)

print()
print("min loss", minLoss, "coefficients", bestCoefficients)

plotData(fileData[0], fileData[1], bestCoefficients)