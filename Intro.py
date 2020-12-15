# -*- coding: utf-8 -*-
"""
Created on Monday Dec 14 09:51:13 2020
@author: WoleOlufayo
"""
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

#
# """Student Maths performance Prediction App"""
data = pd.read_csv("StudentsPerformance.csv", sep=",")

st.write("""
    # Student Math score performance Prediction WebApp
    """)
html_temp = """
    <div style = "background - color: #f0f0f5; padding: 5px">
    <h3 style="color:#666666;text-align:left; line-height: 1.5">
    <p>This Web App will predict student mathematics performance 
    in exams once the following (6) parameters are inputed.<br> 
    This is based on Deep learning algorithms with data from a School in North America.</p></h3>
    Dataset: <br> https://www.kaggle.com/spscientist/students-performance-in-exams<br> 
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.header('Set students Input Parameters')

if st.checkbox('Show Summary of Dataset'):
    st.write(data.describe())


# @st.cache
def inputParameters():
    gender = st.sidebar.radio('1. What is the gender?', ('male', 'female'))
    if gender == 'female':
        gender = 0
    elif gender == 'male':
        gender = 1
    else:
        gender = 2
    race = st.sidebar.radio('2. What is the race?', ('group A', 'group B', 'group C', 'group D', 'group E'))
    if race == 'group A':
        race = 0
    elif race == 'group B':
        race = 1
    elif race == 'group c':
        race = 2
    elif race == 'group D':
        race = 3
    else:
        race = 4
    parent_education = st.sidebar.radio('2. What is the parents education?', (
        'high school', 'some college', 'bachelors degree', 'associates degree', 'masters degree'))
    if parent_education == 'high school':
        parent_education = 2
    elif parent_education == 'some high school':
        parent_education = 5
    elif parent_education == 'some college':
        parent_education = 4
    elif parent_education == 'bachelors degree':
        parent_education = 1
    elif parent_education == 'associates degree':
        parent_education = 0
    elif parent_education == 'masters degree':
        parent_education = 3
    else:
        parent_education = 6
    lunch = st.sidebar.radio('2. Did the student have lunch?', ('free/reduced', 'standard'))
    if lunch == 'free/reduced':
        lunch = 0
    elif lunch == 'standard':
        lunch = 1
    else:
        lunch = 2
    writing = st.sidebar.slider('2. What is test writing score?', 0, 100, 1)
    reading = st.sidebar.slider('2. What is test reading score?', 0, 100, 1)

    features = {'gender': gender, 'race': race,
                'parent_education': parent_education, 'lunch': lunch, 'writing': writing, 'reading': reading}

    feat = pd.DataFrame(features, index=[0])
    return feat


df = inputParameters()
df1 = np.array(df)

st.subheader('User Input parameters')
st.write(df)

from_pickle = open("student_performance.pickle", "rb")
regression_model = pickle.load(from_pickle)

st.text("")


def predictReg():
    prediction = regression_model.predict(df)
    predict = np.around(prediction, 2)
    return float(predict)


performanceGot = predictReg()

st.text("")
if st.button('PREDICT PERFORMANCE'):
    st.write("**Math Score**", performanceGot, " -*based on Deep Learning Algorithm (80% accuracy)*")

url = '[SOURCE CODE](https://github.com/Wolesword/MathScorePrediction-WebApp)'

st.markdown(url, unsafe_allow_html=True)
# st.subheader(')
# Just to show the actual values

# **************************************************************************************************************