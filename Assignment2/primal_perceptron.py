__author__ = 'jtakwani'

import pandas as pd
import numpy as np

'''
This function implements primal form of perceptron algorithm.
Trains the data set until no mistakes i.e. no alpha values change.
It then calls the calculateAccuracy Method.
'''
def primalPerceptron(traindata,trainlabel,testdata,testlabel):
    m,n = traindata.shape
    bias = 0.0
    weight = [0*x for x in range(len(traindata[0]))]

    while True:
        count = 0

        for i in range(m):
            xi = traindata[i]
            yi = trainlabel[i]
            predictedY = np.dot(weight,xi)

            if(predictedY + bias)* yi <= 0:
                count +=1
                bias = bias+yi
                weight = map(sum,zip(weight, [x * yi[0] for x in xi]))
        print count
        if converged(weight,bias,traindata,trainlabel):
            break
    calculateAccuracy(weight,testdata,testlabel,bias)

'''
This function checks if mistkaes occur and returns true
only if there are no mistakes.
'''
def converged(weight,bias,traindata,trainlabel):
    for i in range(len(traindata)):
            xi = traindata[i]
            yi = trainlabel[i]
            predictedY = np.dot(weight,xi)

            if(predictedY + bias)* yi <= 0:
                return False
    return True

'''
this function calulates the accuracy.
'''
def calculateAccuracy(weight,dataMatrix,label,bias):
    count = 0
    result = []
    result.append(0)
    for i in range(1,len(label)):
        ans = np.dot(weight,dataMatrix[i]) + bias
        if ans > 0:
            result.append(1)
        else:
            result.append(-1)
        if result[i] == label[i]:
            count += 1
    print "Accuracy: " + str((count * 100 /len(label)))
    print "Weights: " + str(weight)
    print "Normalized Weights: " + str([w/bias for w in weight])
'''
the program starts here. From the text file we build data frames.
We separate the data in fatures and label. 70% data is used for training.
30% is used for testing.
'''

def main():
    dataFrame = pd.read_table('a2DataSet/perceptron/percep1.txt')
    traindata = np.array(dataFrame.iloc[0:700 , :-1])
    testdata = np.array(dataFrame.iloc[700:, :-1])
    trainlabel = np.array(dataFrame.iloc[0:700 ,-1:])
    testlabel = np.array(dataFrame.iloc[700: ,-1:])
    primalPerceptron(traindata,trainlabel,testdata,testlabel)

if __name__ == "__main__":
    main()