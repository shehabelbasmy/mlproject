#import library

import numpy as np # mathematical functions, 
import pandas as pd #data analysis and manipulation tool, 
import seaborn as sns #data visualization library, 

import matplotlib.pyplot as plt#data visualization library, 
from matplotlib.colors import ListedColormap#data visualization library, 

from sklearn.model_selection import train_test_split #separation of the data set as training and testing , 
from sklearn.preprocessing import RobustScaler #Scale features using statistics that are robust to outliers,
from sklearn.datasets import make_moons, make_circles, make_classification #create datasets 2 binary 1 multiclass, 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier #ensemble learnig , 

#warning library,
import warnings
warnings.filterwarnings('ignore') 

#create datasets
random_state = 42
n_samples = 1000 #number of samples per cluster, 
n_features = 10 #number of features for each sample, 
n_classes = 2 #number of classes (or labels),

#Larger values introduce noise in the labels and make the classification task harder
noise_moon = 0.3
noise_circle = 0.3
noise_class = 0.3 

X,y = make_classification(n_samples = n_samples,
                    n_features = n_features,
                    n_classes = n_classes,
                    n_repeated = 0,#number of duplicated features, 
                    n_redundant = 0,#number of redundant features,
                    n_informative = n_features-1,#number of informative features, 
                    random_state = random_state,
                    n_clusters_per_class = 1,#number of clusters per class, 
                    flip_y = noise_class)


data = pd.DataFrame(X)
data["target"] = y
plt.figure()
sns.scatterplot(x = data.iloc[:,0], y =  data.iloc[:,1], hue = "target", data = data ) #visualization, 

data_classification = (X,y)

moon = make_moons(n_samples = n_samples, noise = noise_moon, random_state = random_state)

#data = pd.DataFrame(moon[0])
#data["target"] = moon[1]
#plt.figure()
#sns.scatterplot(x = data.iloc[:,0], y =  data.iloc[:,1], hue = "target", data = data ) #visualization, 

circle = make_circles(n_samples = n_samples, factor = 0.1,  noise = noise_circle, random_state = random_state)

#data = pd.DataFrame(circle[0])
#data["target"] = circle[1]
#plt.figure()
#sns.scatterplot(x = data.iloc[:,0], y =  data.iloc[:,1], hue = "target", data = data ) #visualization, 
datasets = [moon, circle]
 
# Basic Classifiers : KNN, SVM, DT
n_estimators = 10 #number of trees in the forest, 

svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 15)
dt = DecisionTreeClassifier(random_state = random_state, max_depth = 2)

rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, max_depth = 2)
ada = AdaBoostClassifier(base_estimator = dt, n_estimators = n_estimators, random_state = random_state)
v1 = VotingClassifier(estimators = [('svc',svc),('knn',knn),('dt',dt),('rf',rf),('ada',ada)])

names = ["SVC", "KNN", "Decision Tree", "Random Forest", "AdaBoost", "V1"]
classifiers = [svc, knn, dt, rf, ada, v1]

h=0.2 #resolution, 
i = 1
figure = plt.figure(figsize=(18, 6))
#Training of algorithms and visualization of results

for ds_cnt, ds in enumerate(datasets): #datasets --> circle and moon , ds_cnt --> index
    # preprocess dataset, split into training and test part
    X, y = ds
    X = RobustScaler().fit_transform(X)#Scale features using statistics that are robust to outliers,
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=random_state)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    #meshgrid --> used to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    #colors for visualization 
    cm = plt.cm.RdBu #visualization
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
                                
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i) # len(datasets) : row(satır), len(classifiers) +1 : column(sütun)
    
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k') #visualization, tr:görselleştirelim
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,marker = '^', edgecolors='k') #visualization, tr:görselleştirelim
    
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    print("Dataset # {}".format(ds_cnt))
          
    # classifiers : KNN , SVC , DT
    for name, clf in zip(names, classifiers):
        
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        
        clf.fit(X_train, y_train) 
        
        score = clf.score(X_test, y_test)
        
        print("{}: test set score: {} ".format(name, score))
        
        score_train = clf.score(X_train, y_train)  
        
        print("{}: train set score: {} ".format(name, score_train))
        print()
        
        #The hasattr() method returns true if an object has the given named attribute and false if it does not.
       
        if hasattr(clf, "decision_function"):
            #ravel -->  Return a contiguous flattened array, 
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])#np.c --> concatenation 
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # eng :Put the result into a color plot, 
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        #Plot the training points,
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        #Plot the testing points, 
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,marker = '^',
                   edgecolors='white', alpha=0.6)

        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        score = score*100
        ax.text(xx.max() - .3, yy.min() + .3, ('%.1f' % score),
                size=15, horizontalalignment='right')
        i += 1
    print("-------------------------------------")

plt.tight_layout()
plt.show()

