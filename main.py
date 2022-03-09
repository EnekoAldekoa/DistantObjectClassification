import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix

data = pd.read_csv('./Data/star_classification.csv')

data['class'].value_counts()

data.info()

# Balancear los datos (undersampling)

# data = data.drop(
#    data[data['class'] == 'GALAXY'].sample((data['class'] == 'GALAXY').sum() - (data['class'] == 'QSO').sum()).index)
# data = data.drop(
#    data[data['class'] == 'STAR'].sample((data['class'] == 'STAR').sum() - (data['class'] == 'QSO').sum()).index)

# Quitar columnas que "no siven"

# data = data.drop(
#    ["g", "r", "i", "spec_obj_ID", "obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "MJD", "fiber_ID"], axis=1)

# Train/Test split

x = data.drop(['class'], axis=1).values
y = data.loc[:, 'class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# SMV

# svm_clf = svm.SVC(kernel='rbf', C=1, random_state=0)
# svm_clf.fit(x_train, y_train)
# predicted = svm_clf.predict(x_test)
# score = svm_clf.score(x_test, y_test)
# svm_score_ = np.mean(score)

# print('Accuracy : %.3f' % svm_score_)

# RandomForest


r_forest = RandomForestClassifier()
r_forest.fit(x_train, y_train)
predicted = r_forest.predict(x_test)
score = r_forest.score(x_test, y_test)
rf_score_ = np.mean(score)

print('Accuracy : %.3f' % rf_score_)

classes = ['GALAXY', 'STAR', 'QSO']

r_forest_cm = ConfusionMatrix(r_forest, classes=classes, cmap='GnBu')

r_forest_cm.fit(x_train, y_train)
r_forest_cm.score(x_test, y_test)
r_forest_cm.show()
