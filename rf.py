import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import pickle
import copy
from sklearn.metrics import confusion_matrix
import pandas as pd
import myFuncs
import createData
import functools



break_flag=0
trainingData = createData.trainingData
trainingLabels = createData.trainingLabels
testData = createData.testData
testLabels = createData.testLabels
tempData = copy.deepcopy(trainingData)
tempLabels = copy.deepcopy(trainingLabels)

batch_size = 150000 #from batchSize.py


classifier = pickle.load(open('rfModel_'+str(batch_size)+'.pkl', 'rb'))
print("classifier loaded")

classifier = RandomForestClassifier(n_jobs=-1, n_estimators=400, criterion='entropy', warm_start=True)
for i in range(len(trainingData)// batch_size + 1):
	if (len(np.unique(trainingLabels[i*batch_size:((i+1)*batch_size - 1)])) < 4):
		break_flag=1
		print('unequal class distribution ... quitting')
		break
	classifier.fit(trainingData[i*batch_size:((i+1)*batch_size - 1)], trainingLabels[i*batch_size:((i+1)*batch_size - 1)])
	pickle.dump(classifier, open('rfModel_'+str(batch_size)+'.pkl', 'wb'))
	classifier.n_estimators += 400

if break_flag==0:
	prediction = classifier.predict(trainingData)
	conf_mat_train = confusion_matrix(trainingLabels, prediction)

	prediction = classifier.predict(testData)
	conf_mat_test = confusion_matrix(testLabels, prediction)

	print('creating class maps ..')
	myFuncs.predictClassMap("rfOut_tile1.tif", createData.rasterDS1, classifier)
	myFuncs.predictClassMap("rfOut_tile2.tif", createData.rasterDS2, classifier)
	myFuncs.predictClassMap("rfOut_tile4.tif", createData.rasterDS3, classifier)
	myFuncs.predictClassMap("rfOut_tile7.tif", createData.rasterDS4, classifier)
	myFuncs.predictClassMap("rfOut_tile3.tif", createData.rasterDS5, classifier)
	myFuncs.predictClassMap("rfOut_tile5.tif", createData.rasterDS6, classifier)
	myFuncs.predictClassMap("rfOut_tile6.tif", createData.rasterDS7, classifier)
	myFuncs.predictClassMap("rfOut_tile8.tif", createData.rasterDS8, classifier)
