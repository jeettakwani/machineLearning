__author__ = 'jtakwani'
'''
This program run the multilayer perceptron algorithm
with back propogation. The number of hidden layer(s)
is 1. The data set used for this function is MNIST.
This algorithm only classifies digits 3 and 5.
'''
import numpy as np
import pandas as pd


class MLP:

    '''
    initialize all variables like number of input, number of hidden layers,
    number of output, weights, change in error,activation function.
    '''
    def __init__(self,numbinputs,numbhidden,numboutput):

        self.numberOfInputs = numbinputs +1
        self.numberOfHiddenLayers = numbhidden
        self.numberOfOutput = numboutput

        self.activationInput = np.ones(self.numberOfInputs)
        self.activationHidden = np.ones(self.numberOfHiddenLayers)
        self.activationOutput = np.ones(self.numberOfOutput)

        self.weightInput = np.random.uniform(-0.2,0.2,(self.numberOfInputs,self.numberOfHiddenLayers))
        self.weightOutput = np.random.uniform(-1.0,1.0,(self.numberOfHiddenLayers,self.numberOfOutput))

        self.changeInput = np.zeros((self.numberOfInputs,self.numberOfHiddenLayers))
        self.changeOuput = np.zeros((self.numberOfHiddenLayers,self.numberOfOutput))

        self.dsigmoidfn = np.vectorize(dsigmoid)
        self.sigmoidfn = np.vectorize(sigmoid)

    '''
    classify
    '''
    def neuralNetwork(self,input_records):

        self.activationInput[:self.numberOfInputs-1] = input_records
        self.activationHidden = self.sigmoidfn(np.dot(self.activationInput,self.weightInput))
        self.activationOutput = self.sigmoidfn(np.dot(self.activationHidden,self.weightOutput))

        return self.activationOutput

    '''
    implements back propogation algorithm
    '''
    def backPropogation(self,label,eta,momentum):

        output_errors = dsigmoid(self.activationOutput) * (label - self.activationOutput)


        currentWeight = np.multiply(eta,np.multiply(output_errors,self.activationHidden)).reshape(self.numberOfHiddenLayers,1)


        self.weightOutput = np.add(self.weightOutput,currentWeight)
        self.weightOutput = np.add(self.weightOutput,np.multiply(momentum,self.changeOuput))
        self.changeOuput = currentWeight


        hidden_errors = np.multiply(output_errors,self.weightOutput)
        hidden_deltas = np.multiply(hidden_errors,self.dsigmoidfn(self.activationHidden).reshape(self.numberOfHiddenLayers,1))


        weightDelta = np.multiply(eta,np.multiply(hidden_deltas,self.activationInput).T)

        self.weightInput = np.add(self.weightInput,weightDelta)
        self.weightInput = np.add(self.weightInput,np.multiply(momentum,self.changeInput))
        self.changeInput = weightDelta

        return 0.5*((label-self.activationOutput)**2)

    '''
    trains the model
    '''
    def trainModel(self,trainData,trainLabels,max_iterations=100):
        for iter in range(max_iterations):
            error = 0.0
            for i in range(0,trainData.shape[0]):
                record = trainData[i]
                label = trainLabels[i]
                self.neuralNetwork(record)
                error += self.backPropogation(label,0.5,0.1)
                #print("Error value:{:.5f}".format(error[0]))

    '''
    test the model
    '''
    def testModel(self,testData,testLabels):
        count = 0
        result = []
        result.append(0)
        for i in range(1,testData.shape[0]):
            ans = self.neuralNetwork(testData[i])
            if ans > 0.5:
                result.append(1)
            else:
                result.append(0)
            if result[i] == testLabels[i]:
                count += 1

        print "Accuracy with hidden layers {} is {:.3f}%: "\
            .format(self.numberOfHiddenLayers,(count/float(len(testLabels))*100))

'''
calculates logistic sigmoid
'''
def sigmoid(x):
    return 1/float((1+np.exp(-x)))

'''
differentiation of sigmoid
'''
def dsigmoid(y):
    return y * (1-y)

'''
normalize the data
'''
def normalize(Data):
    return Data/255

'''
converts 3 to 0 and 5 to 1
'''
def convert(labels):
    for i in range(1,len(labels)+1):
        if labels.ix[i] == 3:
            labels.ix[i] = 0
        else:
            labels.ix[i] = 1
    return labels


def main():

    trainDataFrame = pd.read_csv('digits/train.csv')
    testDataFrame = pd.read_csv('digits/test.csv')

    trainData = np.array(normalize(trainDataFrame.iloc[1:, 1:]))
    trainLabel = np.array(convert(trainDataFrame.iloc[1:, 0]))

    testData = np.array(normalize(testDataFrame.iloc[1:, 1:]))
    testLabel = np.array(convert(testDataFrame.iloc[1:, 0]))

    for i in [10,20,30,50,100]:
        print 'nh :' + str(i)
        row,features = np.shape(trainData)
        mlp = MLP(features,i,1)
        mlp.trainModel(trainData,trainLabel)
        mlp.testModel(testData,testLabel)


if __name__ == '__main__':
	main()