# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:31:14 2015

@author: neeraj
"""

import matplotlib.pyplot as plt

from sklearn import svm,metrics
from matplotlib.image import cm
from mnist_database import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics,svm
from matplotlib.image import cm
import pydot,StringIO
from IPython.core.display import Image
import matplotlib.pyplot as plt
from sklearn import cross_validation


def plot_dtree(model,fileName):
    dot_tree_data = StringIO.StringIO()
    tree.export_graphviz(model, out_file = dot_tree_data)
    dtree_graph = pydot.graph_from_dot_data(dot_tree_data.getvalue())
    dtree_graph.write_png(fileName)


db=mnist_database()
(images,labels)=db.get_training_data()
(images1,labels1) = db.get_testing_data(n_estimators =25 ,criterion ="gini",max_depth =20,max_features=150)

classifier = RandomForestClassifier()
#classifier = svm.SVC(kernel = 'rbf')
classifier.fit(images,labels)

expected = labels1
predicted = classifier.predict(images1)


print("Classification report for classifier %s:\n%s\n" % (classifier,metrics.classification_report(expected,predicted)))
#print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted))
print("Accuracy score is %s:" %metrics.accuracy_score(expected,predicted))

importances = classifier.feature_importances_
plt.matshow(importances.reshape(28,28),cmap=plt.cm.hot)
plt.show()

X_train, X_test,y_train,y_test = cross_validation.train_test_split(images,labels,test_size=0.3,random_state=0,cv=5)
clf1 = RandomForestClassifier(criterion='entropy',max_depth=20, max_features=150).fit(X_train,y_train)
clf1.score(X_test,y_test)