# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:15:03 2015

@author: neeraj
"""

import matplotlib.pyplot as plt

from sklearn import svm,metrics
from matplotlib.image import cm
from mnist_database import *

db=mnist_database()
(images,labels)=db.get_training_data()
(images1,labels1) = db.get_testing_data()

classifier = svm.SVC(gamma = 0.001)
classifier.fit(images,labels)

expected = labels1
predicted = classifier.predict(images1)

print("Classification report for classifier %s:\n%s\n" % (classifier,metrics.classification_report(expected,predicted)))
print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted))