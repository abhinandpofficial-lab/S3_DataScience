import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
data = pd.read_csv('Iris.csv')
print("First few rows of dataset:")
print(data.head())
x=data.iloc[:,:4]
y=data.iloc[:,-1]
print("\nFeature data (first 5 rows):")
print(x.head())
print("\nLabels:(first 5 rows):")
print(y.head())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
print("\nTraining features(first 5 rows):")
print(x_train.head())
print("\nTesting features(first 5 rows):")
print(x_test.head())
SC=StandardScaler()
X_train=SC.fit_transform(x_train)
X_test=SC.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors=5)
print(classifier.fit(X_train,y_train))
y_pred=classifier.predict(X_test)
print("\n array",y_pred)
print("\nActual labels:",y_test)
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
print("\nConfusion Matrix:",cm)
print("\nAccuracy:",ac)