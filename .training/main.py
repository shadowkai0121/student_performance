import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from joblib import dump

import re

def to_snake_case(s) -> str:
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    s = re.sub(r'\s+', '_', s).lower()
    return s

df = pd.read_csv('Student_performance_data _.csv')

df = df.drop(columns=["StudentID"])

columns = list(df.columns)
categoric_columns = []
numeric_columns = []

for i in columns:
    if len(df[i].unique()) > 5:
        numeric_columns.append(i)
    else:
        categoric_columns.append(i)
df[numeric_columns] = df[numeric_columns].astype('float64')

label_encoder = LabelEncoder()
df = df.copy()
for column in df[categoric_columns]:
    df[column] = label_encoder.fit_transform(df[column])

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

x = df.drop(columns=['GradeClass', 'GPA', 'Age'])
y = df['GradeClass']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y)

classification_models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(),
}

x_train = x_train[['Absences', 'StudyTimeWeekly', 'ParentalSupport']]
x_test = x_test[['Absences', 'StudyTimeWeekly', 'ParentalSupport']]

for name, clf in classification_models.items():
    model = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    dump(model, "../models/" + to_snake_case(name) + ".pkl")
    print("../models/" + to_snake_case(name) + ".pkl")
