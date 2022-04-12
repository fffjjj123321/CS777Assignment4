from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
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

if __name__ == "__main__":

	# sc = SparkContext(appName="LogisticRegression")

	# Use this code to reade the data
	corpus = sc.textFile(sys.argv[1], 1)
	keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', str(x[1])).lower().split()))


	allWords = keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x:(x,1))

	allCounts = allWords.reduceByKey(lambda x,y:x+y)

	topWords = allCounts.top(10000,key = lambda x:x[1])

	topWordsParal = sc.parallelize(range(10000))
	dictionary = topWordsParal.map (lambda x : (topWords[x][0], x))

	targetWords = dictionary.filter(lambda x:x[0] in ['applicant', 'and', 'attack', 'protein', 'car']).take(5)
	print(targetWords)



