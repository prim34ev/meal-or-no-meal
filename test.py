import numpy as np
import pandas as pd
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

testData = pd.read_csv('test.csv', low_memory=False, header=None)
x = np.asarray(testData)

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
y = loaded_model.predict(x)

y_predict = pd.DataFrame(y)
y_predict.to_csv('Result.csv',index = False, header = None) 
