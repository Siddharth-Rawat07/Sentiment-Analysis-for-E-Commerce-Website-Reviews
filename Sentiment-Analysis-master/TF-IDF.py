# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:29:30 2019

@author: Anvitha
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
text=pd.read_excel('Amazon reviews.xlsx')
text["Sentence"].head()

# Pre-Processing steps
from DataClean import DataCleaner
cleanData = DataCleaner()
sentences , emotions = cleanData.cleanData(text["Sentence"]) , text["Label"]

#split the data into 70% training and 30% test data
from sklearn.model_selection import train_test_split
Sen_train, Sen_test, Label_train, Label_test = train_test_split( sentences , emotions , test_size=0.3, random_state=42, shuffle=True,stratify=emotions)


from DataClean import ModelAnalysis
from replacers import GetStopWords
stopwords = GetStopWords()
print()

#Multinomial Naive Bayes classifier
modelanalysis = ModelAnalysis()
model_NB = MultinomialNB()
modelFit_NB , vectorizer_NB = modelanalysis.VectAndTrans(Sen_train,Label_train,model_NB,stopwords.getStopWords())
#predict how well the Naive Bayes model performs on test data
predictedModel_NB= modelanalysis.PredictEmotion(vectorizer_NB,Sen_test,modelFit_NB)

# import performance metrics to evaluate the Naive Bayes model
from DataClean import PerformanceMetrices
modelMetrices = PerformanceMetrices()
from sklearn.metrics import precision_recall_fscore_support
precision , recall , fbetascore, support = precision_recall_fscore_support(Label_test, predictedModel_NB)
print("Accuracy of Naive Bayes:" +str(modelMetrices.Accuracy(predictedModel_NB,Label_test)))
print("Confusion Matrix of Naive Bayes" +str(modelMetrices.Confusion_matrix(predictedModel_NB,Label_test)))
print("Precision of Naive Bayes:" +str(precision))
print("Recall of Naive Bayes:" +str(recall))
print("F1-score of Naive Bayes:" +str(fbetascore))

#SVM Classifier
from sklearn.svm import LinearSVC
model_SVM = LinearSVC()
modelFit_SVM , vectorizer_SVM = modelanalysis.VectAndTrans(Sen_train,Label_train,model_SVM,stopwords.getStopWords())

# predict how well the SVM model performs on test data
predictedModel_SVM= modelanalysis.PredictEmotion(vectorizer_SVM,Sen_test,modelFit_SVM)

# import performance metrics to evaluate the SVM model
from sklearn.metrics import precision_recall_fscore_support
precision , recall , fbetascore, support = precision_recall_fscore_support(Label_test, predictedModel_SVM)
print("Accuracy of SVM:" +str(modelMetrices.Accuracy(predictedModel_SVM,Label_test)))
print("Confusion Matrix of SVM" +str(modelMetrices.Confusion_matrix(predictedModel_SVM,Label_test)))
print("Precision of SVM:" +str(precision))
print("Recall of SVM:" +str(recall))
print("F1-score of SVM:" +str(fbetascore))

