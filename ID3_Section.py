import sklearn
import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
df1 = pd.read_csv('aihealth.csv')
df1['Sex'].replace({'M' : 0, 'F' : 1}, inplace = True)
df1['ChestPainType'].replace({'ATA' : 0, 'NAP' : 1, 'ASY' : 2, 'TA' : 3}, inplace = True)
df1['RestingECG'].replace({'Normal' : 0, 'ST' : 1, 'LVH' : 2}, inplace = True)
df1['ExerciseAngina'].replace({'N' : 0, 'Y' : 1}, inplace = True)
df1['ST_Slope'].replace({'Flat' : 0, 'Up' : 1, 'Down' : 2}, inplace = True)
df1['HeartDisease'].replace({'FALSE' : 0, 'TRUE' : 1}, inplace = True)
df1 = df1.drop(columns= ['Cholesterol', 'MaxHR', 'FastingBS'])
df1.head()
X = df1.iloc[:, 0:8]
y = df1.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=11, test_size=0.2)

dtc_cart = DecisionTreeClassifier(max_depth=3, criterion='gini')

dtc_cart.fit(X_train, y_train)
y_pred = dtc_cart.predict(X_test)

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
tree = export_graphviz(dtc_cart)
graph = graphviz.Source(tree)
graph.render('Cart')
