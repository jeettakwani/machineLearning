__author__ = 'jtakwani'

import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys

'''
this function implements the Gausian kernel
'''
def nonLinearKernel(xi,xj):
    return np.exp(-0.5*(distance.euclidean(xi,xj)**2))

'''
this function implements the linear kernel
'''
def linearKernel(xi,xj):
    return np.dot(xi,xj)

'''
This function implements dual form of perceptron algorithm.
Trains the data set until no mistakes i.e. no alpha values change.
It then calls the calculateWeight Method.
'''
def dualPerceptron(traindata,trainlabel,testdata,testlabel,option):
    m,n = traindata.shape
    bias = 0.0
    alpha = np.zeros(m,dtype=float)
    while True:
        count = 0

        for i in range(m):
            sum = 0.0
            xi = traindata[i]
            yi = trainlabel[i]

            for j in range(m):
                xj = traindata[j]
                yj = trainlabel[j]
                if option == "1":
                    sum += yj*alpha[j]*linearKernel(xi,xj)
                else:
                    sum += yj*alpha[j]*nonLinearKernel(xi,xj)
            if(sum + bias)* yi <= 0:
                count +=1

                alpha[i] = alpha[i] + 1
                bias = bias+yi
        print count
        if count == 0:
            break
    calculateWeight(traindata,trainlabel,testdata,testlabel,alpha,bias,option)

'''
This function calculates f(x) = alphai*yi*kernel for all i
and checks the value agaisnt label. Depending on the number of
matched label it prints the accuracy.
'''
def calculateWeight(traindata,trainlabel,testdata,testlabel,alpha,bias,option):
    count = 0
    for i in range(len(testdata)):
        xi = testdata[i]
        sum = 0.0
        for j in range(len(traindata)):
            xj = traindata[j]
            labelj = trainlabel[j]
            if option == "1":
                sum += labelj*alpha[j]*linearKernel(xi,xj)
            else:
                sum += labelj*alpha[j]*nonLinearKernel(xi,xj)

        if (sum+bias)*testlabel[i] > 0:
            count += 1

    print ((count/len(testdata))*100)

'''
this function calulates the accuracy.
'''
def calculateAccuracy(weight,dataMatrix,label,bias):
    count = 0
    result = []
    result.append(0)
    for i in range(1,len(label)):
        ans = np.dot(weight.T,dataMatrix[i]) + bias
        if ans > 0:
            result.append(1)
        else:
            result.append(-1)
        if result[i] == label[i]:
            count += 1
    print "Accuracy: " + str(count * 100 /len(label))

'''
the program starts here. From the text file we build data frames.
We separate the data in fatures and label. 70% data is used for training.
30% is used for testing.
'''
def main():
    option = raw_input("Select 1 for linear or 2 for non linear kernel")

    if option not in ("1","2"):
        print "invalid option"
        return

    if option == "1":
        file = 'percep1.txt'
        path = 'a2DataSet/perceptron/'+file
    else:
        file = 'percep2.txt'
        path = 'a2DataSet/perceptron/'+file

    dataFrame = pd.read_table(path)
    traindata = np.array(dataFrame.iloc[0:700 , :-1])
    testdata = np.array(dataFrame.iloc[700:, :-1])
    trainlabel = np.array(dataFrame.iloc[0:700 ,-1:])
    testlabel = np.array(dataFrame.iloc[700: ,-1:])
    dualPerceptron(traindata,trainlabel,testdata,testlabel,option)


if __name__ == "__main__":
    main()