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

# https://www.kaggle.com/spscientist/students-performance-in-exams
# """Student performance Prediction App"""
data = pd.read_csv("StudentsPerformance.csv", sep=",")

st.write("""
    # Student performance Prediction App
    """)
html_temp = """
    <div style = "background - color: #f0f0f5; padding: 5px">
    <h3 style="color:#666666;text-align:left; line-height: 1.5">
    <p>This Web App will predict student performance 
    in exams once the following (5) parameters are inputed.<br> 
    This is based on Deep learning 
    algorithms with data from a School in North America.</p></h3>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.header('Set students Input Parameters')

if st.checkbox('Show Summary of Dataset'):
    st.write(data.describe())

@st.cache
def inputParameters():
    bedrooms = st.sidebar.slider("1. No of bedrooms?", 0, 12, 4)
    # st.write(""" **You selected** """, bedrooms, """**bedrooms**""")

    bathrooms = st.sidebar.slider("2. No of bathrooms?", 0, 15, 5)
    # st.write(""" **You selected** """, bathrooms, """**bathrooms**""")
    sqft_living = st.sidebar.slider("3. Square footage of the house?", 500, 15000, 2000)
    # st.write(""" **You chose** """, sqft_living,"""**Square fts**""")

    sqft_lot = st.sidebar.slider("4. Square footage of the lot?", 500, 170000, 1200)
    # st.write(""" **You wrote** """, sqft_lot,"""**Square fts**""")
    floors = st.sidebar.slider("5. No of floors?", 0, 5, 3)
    # st.write(""" **You selected** """, floors,"""**floors**""")
    # views = st.sidebar.slider("6. No of viewings of the house?", 0,10,0)
    # st.write(""" **You selected** """, views,"""**views**""")
    condition = st.sidebar.slider("7. Overall condition? (1 indicates worn out property and 5 excellent)", 0, 5, 3)
    # st.write(""" **You selected** """, condition,"""**as the overall condition of the house**""")
    grade = st.sidebar.slider("8. Overall grade based on King County grading system? (1 poor ,13 excellent)", 0, 13, 6)
    # st.write(""" **You selected grade** """, grade)
    sqft_above = st.sidebar.slider("9. Square footage above basement?", 200, 12000, 5000)
    # st.write(""" **You chose** """, sqft_abovebsmt,"""**Square fts**""")
    sqft_basement = st.sidebar.slider("10. Square footage of the basement?", 0, 7000, 2500)
    # st.write(""" **You chose** """, sqft_basement,"""**Square fts**""")
    yr_built = st.sidebar.slider("11. Year Built?", 1900, 2019, 2009)
    # st.write(""" **You selected** """, yr_built)
    yr_renovated = st.sidebar.radio('12. Year renovated?', ('Known', 'Unknown'))
    if yr_renovated == 'Unknown':
        yr_renovated = 0
    else:
        yr_renovated = st.sidebar.slider("Year Renovated?", 1900, 2019, 2010)

    zipcode = st.sidebar.slider("13. Zipcode of the house?", 98001, 98288, 98250)
    # st.write(""" **You selected** """, zipcode)
    lat = st.sidebar.slider("14. Location of House (lattitude)?", 47.000100, 47.800600, 47.560053, 0.000001, "%g")
    long = st.sidebar.slider("15. Location of House (longitude)?", -122.6000000, -121.300500, -122.213896, 0.000001,
                             "%g")
    sqft_living15 = st.sidebar.slider("16. Square footage of the house in 2015?", 200, 12000, 3500)
    # st.write(""" **You chose** """, sqft_living15,"""**Square fts**""")
    sqft_lot15 = st.sidebar.slider("17. Square footage of the lot in 2015?", 200, 12000, 3700)
    # st.write(""" **You chose** """, sqft_lot15,"""**Square fts**""")
    waterfront = st.sidebar.radio('18. House has Waterfront View?', ('Yes', 'No'))
    if waterfront == 'Yes':
        waterfront = 1
    else:
        waterfront = 0

    features = {'bedrooms': bedrooms, 'bathrooms': bathrooms,
                'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
                'floors': floors, 'waterfront': waterfront,
                'condition': condition, 'grade': grade,
                'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
                'yr_built': yr_built, 'yr_renovated': yr_renovated, 'zipcode': zipcode,
                'lat': lat, 'long': long, 'sqft_living15': sqft_living15,
                'sqft_lot15': sqft_lot15}

    feat = pd.DataFrame(features, index=[0])

    return feat

df = inputParameters()
df1 = np.array(df)

st.subheader('User Input parameters')
st.write(df)

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

'''
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
'''

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
