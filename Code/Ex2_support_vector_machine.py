# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:15:03 2015

@author: neeraj
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm,metrics
from matplotlib.image import cm
from mnist_database import *
from sklearn.metrics import confusion_matrix

db=mnist_database()
(images,labels)=db.get_training_data()
(images1,labels1) = db.get_testing_data()

def  plot_confusion_matrix(cm,title='Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
   # tick_marks = np.arrange(len(labels))
   # plt.xticks(tick_marks,labels,rotation=45)
    #plt.yticks(tick_marks,labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#classifier1 = svm.LinearSVC()
#classifier2 = svm.SVC(kernel = 'linear')
#classifier3 = svm.SVC(kernel = 'poly')
classifier4 = svm.SVC(kernel = 'rbf')
#classifier1.fit(images[0:1000],labels[0:1000])
#classifier2.fit(images[0:1000],labels[0:1000])
#classifier3.fit(images[0:1000],labels[0:1000])
classifier4.fit(images[0:1000],labels[0:1000])

expected = labels1
#predicted1 = classifier1.predict(images1)
#predicted2 = classifier2.predict(images1)
#predicted3 = classifier3.predict(images1)
predicted4 = classifier4.predict(images1)

#print("Classification report for classifier %s:\n%s\n" % (classifier1,metrics.classification_report(expected,predicted1)))
#print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted1))
#print("Accuracy score is %s:" %metrics.accuracy_score(expected,predicted1))

#print("Classification report for classifier %s:\n%s\n" % (classifier2,metrics.classification_report(expected,predicted2)))
#print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted2))
#print("Accuracy score is %s:" %metrics.accuracy_score(expected,predicted2))




print("Classification report for classifier %s:\n%s\n" % (classifier3,metrics.classification_report(expected,predicted3)))
print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted3))
print("Accuracy score is %s:" %metrics.accuracy_score(expected,predicted3))




print("Classification report for classifier %s:\n%s\n" % (classifier4,metrics.classification_report(expected,predicted4)))
print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted4))
print("Accuracy score is %s:" %metrics.accuracy_score(expected,predicted4))

cm = confusion_matrix(expected,predicted4)
plt.figure()
plot_confusion_matrix(cm)
plt.show()

