__author__ = 'jtakwani'

# This program implements Naive Bayes Algorithm. The program uses pandas and Skleanr python
# libraries. The program starts at the main function. A function named handleData is called.
# This function is used to read the data from the file insert a row with column names and give
# give each column with categorical data as column type 'category'. This columns are then run
# through a pandas function which gives a number to each categorical value in that column. This
# method calls another method call handleMissingValues to handle missing value '?'. To handle
# missing values the function from pandas library called fillna is used and value of the most
# occuring value is used to replace '?'. Once the missing daa is handled to new files are created
# for test and train. Control returns to the main method and the training and test data are divided
# into data frames with all columns and except the class column and 2 more data frames for only the
# class column for test and train data respectively. A model is fit and run through prediction method
# Accuracy is calculated based on the number of matched records.
#
#


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def handleData(filename,flag):
    dataFrame = pd.read_csv(filename, skipinitialspace=True, na_values='?',
                            names= ['AGE','WORKCLASS','FNLWGT', 'EDUCATION',
                                    'EDUCATION-NUM','MARITAL STATUS','OCCUPATION','RELATIONSHIP',
                                    'RACE','SEX','CAPITAL-GAIN','CAPITAL-LOSS','HOURSPW',
                                    'NATIVE-COUNTRY','CLASS'], header = None)

    columns = ['AGE','WORKCLASS','FNLWGT', 'EDUCATION','EDUCATION-NUM','MARITAL STATUS','OCCUPATION','RELATIONSHIP','RACE',
               'SEX','CAPITAL-GAIN','CAPITAL-LOSS','HOURSPW','NATIVE-COUNTRY','CLASS']

    dataFrame = handleMissingValues(dataFrame,columns,flag)

    dataFrame['WORKCLASS'] = dataFrame['WORKCLASS'].astype('category')
    dataFrame['EDUCATION'] = dataFrame['EDUCATION'].astype('category')
    dataFrame['MARITAL STATUS'] = dataFrame['MARITAL STATUS'].astype('category')
    dataFrame['OCCUPATION'] = dataFrame['OCCUPATION'].astype('category')
    dataFrame['RELATIONSHIP'] = dataFrame['RELATIONSHIP'].astype('category')
    dataFrame['RACE'] = dataFrame['RACE'].astype('category')
    dataFrame['SEX'] = dataFrame['SEX'].astype('category')
    dataFrame['NATIVE-COUNTRY'] = dataFrame['NATIVE-COUNTRY'].astype('category')
    dataFrame['CLASS'] = dataFrame['CLASS'].astype('category')

    category_columns = dataFrame.select_dtypes(['category']).columns

    dataFrame[category_columns] = dataFrame[category_columns].apply(lambda x: x.cat.codes)

    creatfile(dataFrame,flag)

def handleMissingValues(dataFrame,columns,flag):
    for c in columns:
        value = dataFrame[c].value_counts().idxmax()

        dataFrame[c].fillna(value,inplace=True)
    return dataFrame

def creatfile(dataFrame,flag):
    if(flag):
        dataFrame.to_csv('a1_datasets/dataset-trai-all-categorical.csv', sep = ",", index = False)
    else:
        dataFrame.to_csv('a1_datasets/dataset-test-all-categorical.csv', sep = ",", index = False)

def main():

   handleData('a1_datasets/census/adult.data', 1)
   handleData('a1_datasets/census/adult.test', 0)

   training = pd.read_csv('a1_datasets/dataset-trai-all-categorical.csv')
   test = pd.read_csv('a1_datasets/dataset-test-all-categorical.csv')

   trainingData = training.iloc[:,0:12]
   trainingTarget = training['CLASS']
   testData = test.iloc[:,0:12]
   testTarget = test['CLASS']
   b = GaussianNB()

   model = b.fit(trainingData,trainingTarget)
   prediction = model.predict(testData)
   count = 0

   for i in range(len(prediction)):
       if prediction[i] == testTarget[i]:
           count = count + 1
   print("Accuracy is " + str(float(count)/float(len(testTarget))*100))
main()