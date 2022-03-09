from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.model_selection import train_test_split

data = pd.read_csv('./Data/star_classification.csv')

data['class'].value_counts()

data.info()

# Balancear los datos (undersampling)

# data = data.drop(
#    data[data['class'] == 'GALAXY'].sample((data['class'] == 'GALAXY').sum() - (data['class'] == 'STAR').sum()).index)

# data.describe()

# data = data.drop(
#   data[data['class'] == 'STAR'].sample((data['class'] == 'STAR').sum() - (data['class'] == 'QSO').sum()).index)

# Quitar columnas que "no siven"

# data = data.drop(
#    ["g", "r", "i", "spec_obj_ID", "obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "MJD", "fiber_ID"], axis=1)

# Train/Test split

x = data.drop(['class'], axis=1)
y = data.loc[:, 'class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Balancear los datos (SMOTE)

sm = SMOTE(random_state=42)
print('Original dataset shape %s' % Counter(y_train))
x_train, y_train = sm.fit_resample(x_train, y_train)
print('Resampled dataset shape %s' % Counter(y_train))

# SVM

svm_clf = svm.SVC(kernel='sigmoid', C=1, random_state=0)
svm_clf.fit(x_train, y_train)
predicted = svm_clf.predict(x_test)
score = svm_clf.score(x_test, y_test)
svm_score_ = np.mean(score)

print('Accuracy : %.3f' % svm_score_)
