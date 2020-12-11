import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
import pickle
import seaborn as sns
from matplotlib import style
import matplotlib.pyplot as plot
import scipy
from scipy import stats
from statsmodels.api import OLS
import streamlit as st


# https://www.kaggle.com/spscientist/students-performance-in-exams

@st.cache
def read_data():
    data = pd.read_csv("StudentsPerformance.csv", sep=",")

    # Interesting info on my data
    # print(data.head(), "\n")
    # print(data.shape, "\n")
    # print(data.dtypes, "\n")
    # print(data.info(), "\n")
    # print(data.race.unique(), "\n")

    en = preprocessing.LabelEncoder()
    data["gender"] = en.fit_transform(list(data["gender"]))
    data["race"] = en.fit_transform(list(data["race"]))
    data["parent_education"] = en.fit_transform(list(data["parent_education"]))
    data["lunch"] = en.fit_transform(list(data["lunch"]))
    data["test_prep"] = en.fit_transform(list(data["test_prep"]))

    # Data frame correlation _Pearson
    '''pd.set_option('display.max_columns', None)
    data_correlation = data[data['gender'] == 1]
    with open("Output.txt", "w") as text:
        print(data_correlation.corr(), file=text)
    '''
    # Another method for Pearson coefficient
    '''data_correlation = data[data['parent_education'] == 1]
    pearson_coef, p_value = stats.pearsonr(data_correlation.gender, data_correlation.race)
    print("Pearson Coeff is: ", pearson_coef)'''

    x = np.array(data[["gender", "race", "parent_education", "lunch", "writing", "reading"]])
    # x = np.array(data[["gender", "race", "parent_education", "lunch", "test_prep"]])
    y = np.array(data[["math"]])
    # y = np.array(data[["math", "reading", "writing"]])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    # print(OLS(y_train, x_train).fit().summary())

    return x_train, y_train, x_test, y_test, x, y


# data reading
x_train, y_train, x_test, y_test, x, y = read_data()

best = 0
for _ in range(1000000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    regression_model = linear_model.LinearRegression()
    regression_model.fit(x_train, y_train)

    score = regression_model.score(x_test, y_test)
    # print(score)

    if score > best:
        best = score

        with open("student_performace.pickle", "wb") as pf:
            pickle.dump(regression_model, pf)
        print(score)


## Recommended system
@st.cache()
def recommended_system(name, num_players):
    from_pickle = open("student_performance.pickle", "rb")
    regression_model = pickle.load(from_pickle)

    results = regression_model.predict(x_test)


# Just to show the actual values

'''

for w in range(len(results)):
    print(np.round(results[w]), y_test[w])'''

# Learning how to plot. but would not work with a multiple regression - works

'''ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(results, hist=False, color="b", label="Fitted Values", ax=ax1)
plot.show()'''

# More plots - histogram of the height - works

'''data.math.plot(kind='hist', color='purple', edgecolor='black', figsize=(10, 7))
plot.title('Distribution of math score', size=24)
plot.xlabel('Math grade', size=18)
plot.ylabel('Frequency', size=18)
plot.show()
'''
# Scatter plot of reading and writing for the male and female - works

'''ax1 = data[data['gender'] == 1].plot(kind='scatter', x='reading', y='writing', color='blue', alpha=0.5, figsize=(10, 7))
data[data['gender'] == 0].plot(kind='scatter', x='reading', y='writing', color='magenta', alpha=0.5, figsize=(10, 7), ax=ax1)
plot.legend(labels=['Males', 'Females'])
plot.title('Relationship between reading and writing', size=24)
plot.xlabel('Reading (hrs)', size=18)
plot.ylabel('Writing (hrs)', size=18);
plot.show()
'''
# Scatter plot of parent education and lunch money by race

'''ax1 = data[data['race'] == 0].plot(kind='scatter', x='reading', y='writing', color='blue', alpha=0.5, figsize=(10, 7))
data[data['race'] == 1].plot(kind='scatter', x='reading', y='writing', color='magenta', alpha=0.5, figsize=(10, 7), ax=ax1)
data[data['race'] == 2].plot(kind='scatter', x='reading', y='writing', color='green', alpha=0.5, figsize=(10, 7), ax=ax1)
data[data['race'] == 3].plot(kind='scatter', x='reading', y='writing', color='yellow', alpha=0.5, figsize=(10, 7), ax=ax1)
data[data['race'] == 4].plot(kind='scatter', x='reading', y='writing', color='red', alpha=0.5, figsize=(10, 7), ax=ax1)
plot.legend(labels=['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
plot.title('Relationship between parent education and wealth', size=24)
plot.xlabel('Parent education', size=18)
plot.ylabel('lunch money', size=18);
plot.show()'''

# Some form of mess here
'''
X_label = 'math'
style.use("ggplot")

# names = ["female", "male"]

plot.scatter(data[X_label], data["math"])
plot.xlabel("Gender of the student")
plot.ylabel("Math score")'''
