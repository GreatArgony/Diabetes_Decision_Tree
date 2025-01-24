import sklearn
import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
df = pd.read_csv('aihealth.csv')
df['Sex'].replace({'M' : 0, 'F' : 1}, inplace = True)
df['ChestPainType'].replace({'ATA' : 0, 'NAP' : 1, 'ASY' : 2, 'TA' : 3}, inplace = True)
df['RestingECG'].replace({'Normal' : 0, 'ST' : 1, 'LVH' : 2}, inplace = True)
df['ExerciseAngina'].replace({'N' : 0, 'Y' : 1}, inplace = True)
df['ST_Slope'].replace({'Flat' : 0, 'Up' : 1, 'Down' : 2}, inplace = True)
df['HeartDisease'].replace({'FALSE' : 0, 'TRUE' : 1}, inplace = True)
df.head()
X = df.iloc[:, 0:11]

y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=11, test_size=0.2)

dtc = DecisionTreeClassifier(max_depth=3, criterion='entropy')

dtc.fit(X_train, y_train)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(dtc, X, y, cv = k_folds)
print(scores)
y_pred = dtc.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
tree = export_graphviz(dtc)
graph = graphviz.Source(tree)
graph.render('ID3')
