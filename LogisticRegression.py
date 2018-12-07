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


def train(dataFileName):
    attributes, dataClass = getData(dataFileName, ',')

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

    return bestTheta, classes


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
    dataFile = open(filename + ".csv", 'r')
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


def choiceTrain():
    # get data file name
    dataFileName = input("Name of datafile, must be CSV? ")
    theta, classes = train(dataFileName)
    thetaFileName = input("Name if file to store theta values in?")
    with open("logRegThetaVals/" + thetaFileName + ".txt", "w+") as f:
        for c in classes:
            f.write(str(c) + ",")
        f.write("\n")
        for set in theta:
            print(set)
            for element in set:
                f.write(str(element) + ",")
            f.write("\n")


def choiceClassify():
    thetaFilename = "logRegThetaVals/" + input("Name of theta values file? ") + ".txt"
    thetaValues = []

    print("Enter the data you would like to classify in the form: 12.7857,2453.65,6855.1")
    print("i.e. float + ',' + float + ',' + float...")

    dataToClassify = input("Enter data: ").strip().split(",")

    with open(thetaFilename, "r") as f:

        data = f.read().split("\n")

        classes = data[0].strip()[:-1].split(",")
        for line in data[1:]:
            if len(line.strip()) == 0:
                continue
            thetaValues.append((line.strip()[:-1]).split(","))

        for i in range(len(classes)):
            classes[i] = int(classes[i])

        for i in range(len(thetaValues)):
            for j in range(len(thetaValues[i])):
                thetaValues[i][j] = float(thetaValues[i][j])


        for i in range(len(dataToClassify)):
            dataToClassify[i] = float(dataToClassify[i])

    print(classes)
    print(thetaValues)
    print(dataToClassify)

    print(classify(classes, thetaValues, dataToClassify))


def menu():
    while True:
        print()
        print("What to do:" )
        print(" - 1: Train on data set?")
        print(" - 2: Classify")
        print(" - 3: Quit")
        choice = input("Enter number of what to do? ")
        if choice == "1":
            # train
            choiceTrain()
        elif choice == "2":
            # classify
            choiceClassify()
        elif choice == "3":
            exit()
        else:
            print("No option selected - try again")


menu()
