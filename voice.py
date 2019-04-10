import sys
import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import essentia.standard as es

import pickle
import matplotlib.pyplot as plt
from ltfatpy import sgram

DATASET    = 'dataset/voice.csv'
PRETRAINED = 'dataset/pretrained.dat'
TESTSET    = 'test/test.csv'

def train():

    mydata = pd.read_csv(DATASET)

    #Plot the histograms

    male = mydata.loc[mydata['label']=='male']
    female = mydata.loc[mydata['label']=='female']

    #Prepare data for modeling

    mydata.loc[:,'label'][mydata['label']=="male"] = 0
    mydata.loc[:,'label'][mydata['label']=="female"] = 1
    mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2)
    scaler = StandardScaler()
    scaler.fit(mydata_train.ix[:,0:20])
    X_train = scaler.transform(mydata_train.ix[:,0:20])
    X_test = scaler.transform(mydata_test.ix[:,0:20])
    y_train = list(mydata_train['label'].values)
    y_test = list(mydata_test['label'].values)

    #Train random forest model

    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

    pickle.dump(forest, open(PRETRAINED, 'wb'))

def loadPretrained():
    return pickle.load(open(PRETRAINED, 'rb'))

def loadTest():

    ret = []

    mydata = pd.read_csv(TESTSET)
    for row in mydata.head().itertuples():
        i = 0
        for r in row:
            if i > 0:
                ret.append(r)
            i=i+1
    return ret

def test(forest, testdata):

    mydata = pd.read_csv(DATASET)
    male = mydata.loc[mydata['label']=='male']
    female = mydata.loc[mydata['label']=='female']
    mydata.loc[:,'label'][mydata['label']=="male"] = 0
    mydata.loc[:,'label'][mydata['label']=="female"] = 1
    mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2)
    scaler = StandardScaler()
    scaler.fit(mydata_train.ix[:,0:20])

    test_scaled = scaler.transform([testdata])
    predict = forest.predict(test_scaled)

    if predict == 0:
        print 'Male'
    else:
        print 'Female'

    '''
    for i in range(0, 23):
        predict = forest.predict([ X_test[i] ])
        print predict
    '''

def plotSgram(s):
    loader = es.MonoLoader(filename=s, downmix = 'mix', sampleRate = 48000)
    audio = loader()
    _ = sgram(np.asarray(audio, dtype=np.float64), 48000., 90.)
    plt.show()

#train()
forest = loadPretrained()
t = loadTest()
test(forest, t)

# plot
#filename = sys.argv[1]
#plotSgram(filename)
