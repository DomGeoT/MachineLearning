import matplotlib.pyplot as plt
import math
import random
import copy
import numpy


def build2dList(innerSize, outSize):
    l = []
    for i in range(outSize):
        inner = []
        for j in range(innerSize):
            inner.append(0)
        l.append(inner)
    return l


def classify(classes, theta, data):
    '''
    :param classes: list of classes from 0, 1, 2 ... k
    :param theta: array of k M dimensional vectors
    :param data: M dimensional vector of data to classify
    :return: array of tuples (class, prob of being in that class given the model)
    '''
    sum = 0
    for b in classes:
        sum += math.e ** dotProduct(data, theta[b])

    classProb = []
    for b in classes:
        classProb.append((b, (math.e ** dotProduct(data, theta[b])) / sum))
    return classProb


def train():
    attributes, dataClass = getData("simpleData.csv", ',')

    learningRate = 0.000005
    classes = list(set(dataClass))  # list of classes, must be 0, 1, 2 ... k

    numAttributes = len(attributes[0]) # num of attributes per datapoint i.e. M
    numClasses = len(classes)
    theta = build2dList(numAttributes, numClasses)  # list of k M-dimensional vectors

    bestTheta = []
    minLoss = 99999999999

    for i in range(numClasses):
        for j in range(numAttributes):
            theta[i][j] += float(random.randint(0,1000))/1000

    for n in range(100):
        print("Iteration:", n)

        loss = 0.0
        dLoss = build2dList(numAttributes, numClasses)
        for i in range(len(attributes)): # for every datapoint in attributes

            logSumExpValues = []
            for a in classes:
                logSumExpValues.append(dotProduct(attributes[i], theta[a]))

            dp = dotProduct(attributes[i], theta[dataClass[i]])
            lsv = logSumExp(logSumExpValues)

            loss +=  dp - lsv

            for a in classes:
                for k in range(numAttributes):

                    indicator = 0
                    if a == dataClass[i]:
                        indicator = 1

                    sum = 0
                    for b in classes:
                        sum += numpy.power(math.e, dotProduct(attributes[i], theta[b]))
                    probClassGivenModel = numpy.power(math.e, dotProduct(attributes[i], theta[dataClass[i]])) / (sum)

                    dLoss[a][k] += attributes[i][k] * (indicator - probClassGivenModel)

        for a in classes:
            for i in range(numAttributes):
                theta[a][i] += learningRate * dLoss[a][i]

        if math.fabs(loss) < minLoss:
            bestTheta = copy.deepcopy(theta)
            minLoss = math.fabs(loss)
            print("NEW BEST THETA", bestTheta)

    return bestTheta


def dotProduct(A, B):
    sum = 0
    for i in range(len(A)):
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
        dataClass[i] = int(dataClass[i])

    return attributes, dataClass


def logSumExp(values):
    sum = 0.0
    for i in values:
        sum += numpy.power(math.e, i)
    return numpy.log(sum)

#print(train())


def plotData(attributes, dataClass):
    for i in range(len(attributes)):
        if dataClass[i] == 0:
            plt.plot(attributes[i][0], attributes[i][1], 'r+')
        else:
            plt.plot(attributes[i][0], attributes[i][1], 'b+')

    plt.show()


#[[10.379879254987287, 0.558891480372253], [4.873879254987285, 5.573891480372253]]

a, c = getData("simpleData.csv", ',')
print(classify([0,1], [[0.5192223006740697, 0.6349233839829894], [0.3450323006740697, 0.6919683839829895]], [20,20]))
plotData(a,c)

