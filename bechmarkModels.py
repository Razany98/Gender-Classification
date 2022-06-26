#Comparison between different classifiers 

#Accuracy of logstic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
lr_accuracy = classifier.score(X_test, y_test)
print(f"Accuracy of Logistic Regression Classifier is:{lr_accuracy}")

#Accuracy of SVM
from sklearn.svm import SVC
classifier =  SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
lr_accuracy = classifier.score(X_test, y_test)
print(f"Accuracy of SVM Classifier is:{lr_accuracy}")

#Try incresing the regularization of SVM
from sklearn.svm import SVC
model = SVC(C=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#Try applying gamma on SVM 
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)

#Decision Tree

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm) #Print the matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

y_pred = model.predict(X_test)

#Evaluation Matrixes

#predicting the Test set results 
Y_pred = classifier.predict(X_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, Y_pred)
print(cm)

# Use score method to get accuracy of model
score = classifier.score(X_test, y_test)
print(score)

import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


