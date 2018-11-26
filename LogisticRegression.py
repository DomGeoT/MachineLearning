import numpy
import matplotlib.pyplot as plt
import math

global theta
global numClasses
global numAttributes

numClasses = 2
numAttributes = 2

def classifier(X):
    global theta
    '''
    :param X: data
    :param pheta:
    :return: p(y = 1 | X, pheta)
    '''
    x = [1]
    x.extend(X)
    return (math.e ** dotproduct(x, theta)) / (1 + math.e ** dotproduct(x, theta))

def dotproduct(A, B):
    sum = 0
    for i in range(len(A)):
        print(A[i])
        print(B[i])
        sum += A[i] * B[i]
    return sum

def getData(filename, separator):
    '''
    columns 0 - n-2: Attributes
    column n-1:      Class

    :return:
    '''
    dataFile = open(filename, 'r')
    attributes = []
    dataClass = []

    for line in dataFile:
        splitData = line.strip().split(separator)
        if len(line) == 0:
            continue
        attributes.append(splitData[:-1])
        dataClass.append(splitData[-1])

    for i in range(0, len(attributes)):
        d = attributes[i]
        for j in range(len(d)):
            d[j] = float(d[j])
        attributes[i] = d
        dataClass[i] = float(dataClass[i])

    return attributes, dataClass


def setupTheta(classData):
    global theta
    global numAttributes
    global numClasses

    numAttributes = len(attributes[0]) + 1
    numClasses = len(set(classData))

    theta = [[0] * (numAttributes + 1)] * numClasses

def learn(trainingData):
    pass


def plotData(attributes, dataClass):
    for i in range(len(attributes)):
        if dataClass[i] == 0:
            plt.plot(attributes[i][0], attributes[i][1], 'r+')
        else:
            plt.plot(attributes[i][0], attributes[i][1], 'b+')

    p = numpy.linspace(0, 100)

    #pheta = [200, -1.8, 1.5]
    y = []
    for i in range(len(p)):
        y.append((theta[1] * p[i] + theta[0]) / theta[2])

    plt.plot(p, y)


    plt.show()



attributes, dataClass = getData("data2class.csv", ',')
setupTheta(dataClass)
print(theta)

plotData(attributes, dataClass)
