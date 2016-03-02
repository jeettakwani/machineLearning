__author__ = 'jtakwani'

import os
import numpy as np
import random
import pandas as pd
import math
from operator import add
from sklearn.svm import LinearSVC


def Smo(data, classLabel, TestData, TestLabel, constant, tolerance, maxIterations):
    dataMatrix = np.array(data)
    testDataMatrix = np.array(TestData)
    testLabel = np.array(TestLabel)
    label = np.array(classLabel)
    m,n = np.shape(dataMatrix)
    alpha = np.zeros(m)
    bias = 0.0
    iterations = 0
    count = 0

    while(iterations < maxIterations):

        alphas_changed = 0

        for i in range(1,m):

            Ei = float(evaluate(dataMatrix,label,alpha,bias,i)) - float(label[i])

            if((label[i] * Ei < -tolerance and alpha[i] < constant) or
                   (label[i] * Ei > tolerance and alpha[i] > 0)):

                j = i
                while j==i:
                    j = int(np.random.uniform(0,m))

                Ej = float(evaluate(dataMatrix,label,alpha,bias,j)) - float(label[j])

                alphaI_old = alpha[i]
                alphaJ_old = alpha[j]

                if(label[i] != label[j]):
                    L = max(0, alpha[j] - alpha[i])
                    H = min(constant, constant + (alpha[j] - alpha[i]))
                else:
                    L = max(0, (alpha[j] + alpha[i]) - constant)
                    H = min(constant, alpha[j] + alpha[i])

                if L == H:
                    #print"L = H"
                    continue

                eta = 2.0 * (np.dot(dataMatrix[i],dataMatrix[j])) \
                      - (np.dot(dataMatrix[i],dataMatrix[i])) \
                      - (np.dot(dataMatrix[j],dataMatrix[j]))

                if eta >= 0:
                    print "eta is bigger"
                    continue

                alpha[j] -= float(label[j] * (Ei - Ej)/eta)

                if alpha[j] >= H:
                    alpha[j] = H
                elif alpha[j] <= L:
                    alpha[j] = L
                else:
                    alpha[j] = alpha[j]

                if abs(alpha[j] - alphaJ_old) < 0.00001:
                    continue

                alpha[i] = alphaI_old + (label[i]*label[j]*(alphaJ_old - alpha[j]))

                #updating bias
                bias1 = bias - Ei - (label[i]*(alpha[i] - alphaI_old) * \
                                    (np.dot(dataMatrix[i],dataMatrix[i])) - \
                        label[j]*(alpha[j] - alphaJ_old) * (np.dot(dataMatrix[i],dataMatrix[j])))

                bias2 = bias - Ej - (label[i]*(alpha[i] - alphaI_old) * \
                                    (np.dot(dataMatrix[i],dataMatrix[j])) - \
                        label[j]*(alpha[j] - alphaJ_old) * (np.dot(dataMatrix[j],dataMatrix[j])))

                if 0 < alpha[i] < constant:
                    bias = bias1
                elif 0 < alpha[j] < constant:
                    bias = bias2
                else:
                    bias = (bias1 + bias2)/2

                if math.isnan(bias):
                    count += 1
                    print "count" + str(count)

                alphas_changed += 1


                '''
                print "iter: %d i:%d, pairs changed %d" % (
                    iterations, i, alphas_changed
                )
                '''

        if alphas_changed == 0:
            iterations += 1
        else:
            iterations = 0

        #print iterations

    #print alpha
    #print bias

    calculateWeightVector(dataMatrix,label,alpha,bias,testDataMatrix,testLabel)

def evaluate(dataMatrix,label,alpha,bias,i):

    return float(np.dot(np.multiply(alpha,label),(np.dot(dataMatrix,dataMatrix[i].T))) + bias)


#normalize in the range of 0 and 1
def normalizeData(data):

    return data/255

def calculateWeightVector(dataMatrix,label,alpha,bias,testDataMatrix,testLabel):
    weight = [0*x for x in range(len(dataMatrix[0]))]
    for i in range(1,len(dataMatrix)):
        xi = dataMatrix[i]
        result = [label[i]*alpha[i]*x for x in xi]
        weight = np.array(map(add,result,weight))
    #print len(weight)
    calculateAccurace(testDataMatrix,testLabel,alpha,weight,bias)

def calculateAccurace(dataMatrix,label,alpha,weight,bias):
    count = 0
    result = []
    result.append(0)
    for i in range(1,len(label)):
        #print dataMatrix[i]
        ans = np.dot(weight.T,dataMatrix[i]) + bias
        if ans > 0:
            result.append(1)
        else:
            result.append(-1)
        if result[i] == label[i]:
            count += 1
    print (count * 100 /len(label))

def convert(labels):
    for i in range(1,len(labels)+1):
        if labels.ix[i] == 3:
            labels.ix[i] = -1
        else:
            labels.ix[i] = 1
    return labels
def main():

    dataFrame = pd.read_csv('../Data/a2DataSet/digits/train.csv')
    testdataFrame = pd.read_csv('../Data/a2DataSet/digits/test.csv')
    Data = dataFrame.iloc[1:2000, 1:]
    TestData = testdataFrame.iloc[1:, 1:]
    TestLabel = convert(testdataFrame.iloc[1:,0])
    Class = convert(dataFrame.iloc[1:2000, 0])
    Data = normalizeData(Data)
    TestData = normalizeData(TestData)
    #print Data
    Smo(Data, Class, TestData, TestLabel, 0.01, 0.5, 5)


if __name__ == "__main__":
    main()