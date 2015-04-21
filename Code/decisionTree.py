from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm,metrics
from matplotlib.image import cm
from mnist_database import *
import pydot, StringIO
from IPython.core.display import Image
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import cross_validation

def plot_dtree(model,fileName):
    dot_tree_data = StringIO.StringIO()
    tree.export_graphviz(model, out_file = dot_tree_data)
    dtree_graph = pydot.graph_from_dot_data(dot_tree_data.getvalue())
    dtree_graph.write_png(fileName)


db = mnist_database()
(images,labels)=db.get_training_data()
(images1,labels1) = db.get_testing_data()


clf = DecisionTreeClassifier(criterion='entropy',max_depth=20, max_features=350)

clf.fit(images,labels,sample_weight=None,check_input=True)
expected = labels1
predicted = clf.predict(images1)
fileName = 'dtree_gini.png'
plot_dtree(clf,fileName)
Image(filename = fileName)

print("Classification report for classifier %s:\n%s\n" % (clf,metrics.classification_report(expected,predicted)))
print("Accuracy score is %s:" %metrics.accuracy_score(expected,predicted))
#print ("Confusion matrix : \n%s" % metrics.confusion_matrix(expected,predicted))

importances = clf.feature_importances_
#importances = importances.reshape(images[0].shape(28,28))
plt.matshow(importances.reshape(28,28),cmap=plt.cm.hot)
#plt.imshow(images[i].reshape(28,28),cmap=cm.gray)
plt.show()


#cross validation
X_train, X_test,y_train,y_test = cross_validation.train_test_split(images,labels,test_size=0.4,random_state=0)
clf1 = DecisionTreeClassifier(criterion='entropy',max_depth=12, max_features=150).fit(X_train,y_train)
scores = clf1.score(X_test,y_test)

clf1 = DecisionTreeClassifier(criterion='entropy',max_depth=12, max_features=150)
scores = cross_validation.cross_val_score(clf1,images,labels,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))