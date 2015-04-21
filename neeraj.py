from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
import matplotlib.pyplot as plt
from matplotlib.image import cm
from mnist_database import *
#db=mnist_database()
#(images,labels)=db.get_training_data()
#(images1,labels1) = db.get_testing_data()

#implot = plt.imshow(images[2].reshape(28,28),cmap=cm.gray)
#plt.show()
#implot = plt.imshow(images1[56].reshape(28,28),cmap=cm.gray)
#plt.show()
#clf = sklearn.svm.LinearSVC()
#clf.fit(images[0:1000],labels[0:1000])
#print(clf.predict(images1[2]))
#counter = 1
#for i in range(0,10):
 #       plt.subplot(5,5,counter)
  #      plt.imshow(images[i].reshape(28,28),cmap=cm.gray)
   #     counter +=1
#plt.show()

#clf = sklearn.svm.SVC()
#clf.fit(images[0:1000],labels[0:1000])
#counter  = 1
#for i in range(0,10):
 #   print(clf.predict(images1[i]))
  #  plt.subplot(5,5,counter)
   # plt.imshow(images[i].reshape(28,28),cmap=cm.gray)
    #counter+=1
#plt.show()

if __name__ == '__main__':
    db=mnist_database()
    (images,labels)=db.get_training_data()
    (images1,labels1) = db.get_testing_data()
    

    pipeline = Pipeline([('clf',SVC(kernel='rbf',gamma=0.01,C=100))])
    parameters = {
    'clf__gamma': (0.01,0.03,0.1,0.3,1),
    'clf__C': (0.1,0.3,1,3,10,30)

    }
    grid_search = GridSearchCV(pipeline,parameters,n_jobs=2,verbose=1,scoring='accuracy')
    grid_search.fit(images[:10000],labels[:10000])
    print 'Best score: %0.3f' %grid_search.best_score_
    print 'Best paraeters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name,best_parameters[param_name])
        predictions = grid_search.predict(images1)
        print classification_report(labels1,predictions)
 
    

