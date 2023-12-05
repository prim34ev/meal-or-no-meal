import numpy as np
import pandas as pd
import datetime
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

cgmData_file_1 = pd.read_csv('CGMData.csv', sep=',', low_memory = False)
cgmData_file_2 = pd.read_csv('CGM_patient2.csv', sep=',', low_memory = False)
cgm = pd.concat([cgmData_file_1, cgmData_file_2],axis=0)
cgm['dateTime'] = pd.to_datetime(cgm['Date'] + ' ' + cgm['Time'])
cgm = cgm.sort_values(by='dateTime',ascending=True)

insulinData_file_1 = pd.read_csv('InsulinData.csv', sep=',', low_memory = False)
insulinData_file_2 = pd.read_csv('Insulin_patient2.csv', sep=',', low_memory = False)
insulin = pd.concat([insulinData_file_1, insulinData_file_2],axis=0)
insulin['dateTime'] = pd.to_datetime(insulin['Date'] + ' ' + insulin['Time'])
insulin = insulin.sort_values(by='dateTime',ascending=True)

mealTimes = insulin.loc[insulin['BWZ Carb Input (grams)'] > 0][['Index', 'Date', 'Time', 'BWZ Carb Input (grams)', 'dateTime']]
mealTimes['diff'] = mealTimes['dateTime'].diff(periods=1)
mealTimes['shiftUp'] = mealTimes['diff'].shift(-1)

mealTimes = mealTimes.loc[(mealTimes['shiftUp'] > datetime.timedelta (minutes = 120)) | (pd.isnull(mealTimes['shiftUp']))]

cgmdata_withMeal = pd.DataFrame()
for i in range(len(mealTimes)) : 
    preMealTime = mealTimes['dateTime'].iloc[i] - datetime.timedelta(minutes = 30)
    endMealTime = mealTimes['dateTime'].iloc[i] + datetime.timedelta(minutes = 120)
    filteredcgmdata = cgm.loc[(cgm['dateTime'] >= preMealTime) & (cgm['dateTime'] < endMealTime )]
    arr = []
    for j in range(len(filteredcgmdata)) :
        arr.append(filteredcgmdata['Sensor Glucose (mg/dL)'].iloc[j])
    cgmdata_withMeal = cgmdata_withMeal.append(pd.Series(arr), ignore_index=True)

cgmdata_withMeal

no_of_rows= cgmdata_withMeal.shape[0]
no_of_columns = cgmdata_withMeal.shape[1]
cgmdata_withMeal.dropna(axis=0, how='all', thresh=no_of_columns/4, subset=None, inplace=True)
cgmdata_withMeal.dropna(axis=1, how='all', thresh=no_of_rows/4, subset=None, inplace=True)
cgmdata_withMeal.interpolate(axis=0, method ='linear', limit_direction ='forward', inplace=True)
cgmdata_withMeal.bfill(axis=1,inplace=True)
cgmdata_withMeal['label'] = 1

no_meal_time = []
for i in range(len(mealTimes)) : 
    startTime = mealTimes['dateTime'].iloc[i] + datetime.timedelta(minutes = 120)
    endTime = startTime + datetime.timedelta(minutes = 120)
    fullDataEndTime = insulin['dateTime'].iloc[-1]
    no_meal_continue = True
    while (no_meal_continue == True) :
        tempRange = insulin.loc[(insulin['dateTime'] >= startTime) & (insulin['dateTime'] < endTime) & (insulin['BWZ Carb Input (grams)'] > 0)]
        if (len(tempRange) > 0) :
            no_meal_continue = False
        else :
            no_meal_time.append(startTime)
        startTime = startTime + datetime.timedelta(minutes = 120)
        endTime = endTime + datetime.timedelta(minutes = 120)
        if (startTime > fullDataEndTime) :
            no_meal_continue = False

cgmdata_noMeal = pd.DataFrame()
for i in no_meal_time:
    noMealStartTime = i
    noMealEndTime = i + datetime.timedelta(minutes = 120)
    filteredcgmdata = cgm.loc[(cgm['dateTime'] >= noMealStartTime) & (cgm['dateTime'] < noMealEndTime)]
    arr = []
    for j in range(len(filteredcgmdata)):
        arr.append(filteredcgmdata['Sensor Glucose (mg/dL)'].iloc[j])
    if (len(arr) > 24):
        continue
    cgmdata_noMeal = cgmdata_noMeal.append(pd.Series(arr), ignore_index=True)
    
no_of_rows= cgmdata_noMeal.shape[0]
no_of_columns = cgmdata_noMeal.shape[1]
cgmdata_noMeal.dropna(axis=0, how='all', thresh=no_of_columns/4, subset=None, inplace=True)
cgmdata_noMeal.dropna(axis=1, how='all', thresh=no_of_rows/4, subset=None, inplace=True)
cgmdata_noMeal.interpolate(axis=0, method ='linear', limit_direction ='forward', inplace=True)
cgmdata_noMeal.bfill(axis=1,inplace=True)
cgmdata_noMeal['label'] = 0

totalResult = pd.concat([cgmdata_withMeal, cgmdata_noMeal], sort = False)
totalResult = totalResult.interpolate(axis = 0)
condense_totalResult = totalResult[totalResult.columns[:24]]

x = np.asarray(condense_totalResult)
y = np.asarray(totalResult['label'])
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
