import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import os
import pandas as pd
import matplotlib.pyplot as plt
import math as ma
import time

t1 = time.time()

mnist_data = pd.read_csv(os.getcwd()+'/train.csv')
os.getcwd()

#Full data
X = mnist_data[mnist_data.columns[1:]].values
y = mnist_data.label.values

#PCA transformation
pca = PCA()
pca.fit(X)

new_X = pca.transform(X)

n_train = int(2*X.shape[0]/3)

#Trainning data
X_train = X[:n_train, :]
y_train = y[:n_train]

#Testing Data
X_test = X[n_train:, :]
y_test = y[n_train:]

def Display (n):
    #Displaying few numbers of the set
    n_display = 3
    for i in range (n_display):
        plt.subplot(n_display, 1, i+1)
        image = X[i]
        image = image.reshape(28, 28)
        plt.imshow(image)
        plt.text(5, 3, str(y[i]), bbox={'facecolor': 'white', 'pad': 10})
    plt.show()

#Test
t2 = time.time()
print('Step 1 execution time: ' + str(ma.floor(t2-t1)) + 's')

#Full Cross Validation
random_forest = RandomForestClassifier(n_estimators = 100)
cv = cross_val_score(random_forest, X, y, cv = 3, verbose = True)
print('Result of Cross Validation on Random Forest: ' + str(cv.mean()))

#Test
t3 = time.time()
print('Step 2 execution time: ' + str(ma.floor(t3-t2)) + 's')

#96% compression Cross Validation
cvc = cross_val_score(random_forest, new_X[:, range(0,30)], y, cv = 3, verbose = True)
print('Result of Cross Validation on Random Forest: ' + str(cvc.mean()))

#Test
t4 = time.time()
print('Step 3 execution time: ' + str(ma.floor(t4-t3)) + 's')

#Trainning and test by the random forest method
random_forest.fit(X_train, y_train)
random_forest.score(X_test, y_test)

#Test
t5 = time.time()
print('Step 4 execution time: ' + str(ma.floor(t5-t4)) + 's')

#Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_test, random_forest.predict(X_test)), index = range(0,10), columns = range(0,10))
print(cm)

#Test
t6 = time.time()
print('Step 5 execution time: ' + str(ma.floor(t6-t5)) + 's')
print('Total execution time: ' + str(ma.floor(t6-t1)) + 's')

if __name__ == '__main__':
#    print(mnist_data.head()) #Dataset display
    print(X.shape)

#Prediction of an image
print(random_forest.predict(X_test[1].reshape(1, -1)), str(y_test[1]))
