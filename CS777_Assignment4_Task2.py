from __future__ import print_function

import re
import sys
import numpy as np
import pandas as pd
from operator import add
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
import pyspark.ml.feature as ft

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# If needed, use this helper function
# You can implement your own version if you find it more appropriate 
def freqArray (listOfIndices, numberofwords):
	returnVal = np.zeros (10000)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	returnVal = np.divide(returnVal, numberofwords)
	return returnVal


def buildArray(listOfIndices):
    returnVal = np.zeros(10000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    mysum = np.sum(returnVal)
    returnVal = np.divide(returnVal, mysum)
    return returnVal

if __name__ == "__main__":
	
	regex = re.compile('[^a-zA-Z]')
	
	training_corpus = sc.textFile(sys.argv[1], 1)
	testing_corpus = sc.textFile(sys.argv[2], 2)
	training_keyAndText = training_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	testing_keyAndText = testing_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

	regex = re.compile('[^a-zA-Z]')
	training_keyAndListOfWords = training_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	testing_keyAndListOfWords = testing_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

	allWords = training_keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x:(x,1))
	allCounts = allWords.reduceByKey(lambda x,y:x+y)
	topWords = allCounts.top(10000,key = lambda x:x[1])
	topWordsParal = sc.parallelize(range(10000))
	dictionary = topWordsParal.map (lambda x : (topWords[x][0], x))
	
	allWordsWithDocID = training_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
	allDictionaryWords = allWordsWithDocID.join(dictionary)
	justDocAndPos = allDictionaryWords.map(lambda x:x[1])

	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
	allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0],buildArray(x[1])))

	zeroOrOne = allDocsAsNumpyArrays.map(lambda x: [x[0],np.where(x[1] > 0,1,0)])

	numberOfLines = training_keyAndListOfWords.count()
	dfArray = zeroOrOne.map(lambda x:x[1]).treeAggregate(np.zeros(10000),lambda x1, x2:np.add(x1,x2),lambda x1, x2:np.add(x1,x2),3)
	idfArray = np.log(np.divide(np.full(10000, numberOfLines),dfArray))

	numberOfDocs = training_keyAndListOfWords.map(lambda x:('Australian legal case' if x[0][:2] == 'AU' else 'Wikipedia documents',1)).aggregateByKey(0,lambda x,y:np.add(x,y),lambda x,y:np.add(x,y)).take(2)
	numberOfDocs = pd.DataFrame(numberOfDocs)
	numberOfDocs.columns = ['class','numbers']
	featureRDD = allDocsAsNumpyArrays.map(lambda x: (x[0],np.multiply(x[1], idfArray),1 if x[0][:2] == 'AU' else 0))
	featureRDD.cache()

	AUFeatureRDD = featureRDD.filter(lambda x:x[2]==1)
	WikiFeatureRDD = featureRDD.filter(lambda x:x[2]==0)
	AUnumberOfDocs = float(numberOfDocs[numberOfDocs['class']=='Australian legal case']['numbers'])
	WikinumberOfDocs = float(numberOfDocs[numberOfDocs['class']=='Wikipedia documents']['numbers'])

	AUFeatureRDD.cache()
	WikiFeatureRDD.cache()

	test_allWordsWithDocID = testing_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
	test_allDictionaryWords = test_allWordsWithDocID.join(dictionary)
	test_justDocAndPos = test_allDictionaryWords.map(lambda x:x[1])
	test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()
	test_allDocsAsNumpyArrays = test_allDictionaryWordsInEachDoc.map(lambda x: (x[0],buildArray(x[1])))
	# test_featureRDD = test_allDocsAsNumpyArrays.map(lambda x: (x[0],np.multiply(x[1], idfArray),1 if x[0][:2] == 'AU' else 0))
	test_featureRDD = allDocsAsNumpyArrays.map(lambda x: (np.array(x[1]), np.where(x[0][:2] == 'AU', 1, 0)))



	parameter_vector = np.ones(10000)/100
	maxIteration=100
	iter=0
	featureRDD.cache()
	precision=0.00001
	regularization=0.01   
	learning_rate=0.0001
	prev_cost=0
	while iter<maxIteration:
	    res = featureRDD.treeAggregate((np.zeros(10000), 0, 0),lambda x, y:(x[0] + (y[1]) * (-y[0] + (np.exp(np.dot(y[1], parameter_vector))/(1 + np.exp(np.dot(y[1], parameter_vector))))), x[1] + y[0] * (-(np.dot(y[1], parameter_vector))) + np.log(1 + np.exp(np.dot(y[1],parameter_vector))),  x[2] + 1),lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2]))
	    #without regularization
	    cost =  res[1] + regularization * (np.square(parameter_vector).sum())
	    LLH=-res[1]
	    gradient_derivative = (1.0 / res[2]) * res[0] + 2 * regularization * parameter_vector
	    prev_vector = parameter_vector
	    prev_cost=cost
	    parameter_vector = parameter_vector - learning_rate * gradient_derivative
	    if cost>prev_cost:
	            learning_rate=0.5*learning_rate
	    else:
	            learning_rate=1.05*learning_rate
	    if np.sum((parameter_vector - prev_vector)**2)**0.5 < precision:
	            print("Stoped at iteration", iter)
	            break
	    if iter%2==0:
	        print("Iteration No.", iter, " Cost=", cost)
	    iter+=1

	sc.stop()















