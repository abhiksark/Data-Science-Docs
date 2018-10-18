#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:27:56 2017

@author: abhik

Basics functions for Handeling pandas

(Data Preprocessing)

"""

#we usually input csv in dataframes using pandas library 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values




##################################################################################################
"""Preprocessing"""
df.isna().sum() #counting missing data 
df.notnull().all() #removing all the not nulls
df.dropna(how='any') #using any startegy
#if in the dataframe can interpolate using this script and spline = polynomial
df = df.interpolate(method='spline', order=2)
mean = df['coloumn'].mean()
df['coloumn'].fillna(mean,inplace= True)

df.duplicated() #finding duplicates
df.drop_duplicates(inplace=True) #removing duplicates as well

#we should always convert datetime to datetime object rather than string
##################################################################################################
"""Basic Operations"""

df.info() #basic info 

df.shape   #total number of rows and columns
df.dtypes #this returns the datatypes of the columns

df.tail() #if you want to see last few rows then use tail command (default last 5 rows will print)

df[2:5] 

df.columns #names of columns as a list  

df.index 


df.column #priting particular column data 
df['column'] #df.column (both are same) It returns pandas.core.series.Series type
df[['column']] #It returns pandas.core.frame.DataFrame
df[['column1', 'column2']] #getting two or more column at once 
df['column'].max()
df['column'].min() 
df['column'].describe()
df['column'].mean()
df['column'].std()
df['column'].median()
df['column'].quantile(q) # q ~ 0 / 1 
df['column'].unique()
df.nunique() # this returns the number of unique values in each column


df[df.column == df.column.max()]  #select rows which has maximum value

df['salt']['Jan']#Indexing using square brackets
df.eggs['Mar'] #Using column a!ribute and row label
df.loc['May', 'spam'] #here may and spam are row and columns and as it's name it's loc
df.iloc[4, 2] #same thing as above but numbers
pd.merge(df1, df2, on="movie_title")

##################################################################################################
"""Pandas Series"""

"""
pandas Data Structures
Key building blocks
● Indexes: Sequence of labels
● Series: 1D array with Index
● DataFrames: 2D array with Series as columns

"""
prices = [10.70, 10.86, 10.74, 10.71, 10.79]
shares = pd.Series(prices)
print(shares)
"""
Output
0    10.70
1    10.86
2    10.74
3    10.71
4    10.79 
"""

data = df['coloumn']
type(data) # pandas.core.series.Series
newNumpyData = data.values #converts Series into Numpy array

users['fees'] = 0 #Broadcasts to entire column
df.salt > 60 #creates a boolean series
df[df.salt > 60] #Filtering with a Boolean Series
df.eggs[df.salt > 55] += 5  #Modifying a column based on another
df.apply(fun) #apply a func. Where func is passed as parameter
df.apply(lambda n: n//12)
df.map(df1) #mapping dataframe from df to df1

##################################################################################################
"""Building DataFrames"""

#method 1
df = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
'visitors': [139, 237, 326, 456],
'signups': [7, 12, 3, 5] 
}


users = pd.DataFrame(df)



df.reset_index(drop=True, inplace=True)


#method 2
cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
signups = [7, 12, 3, 5]
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
visitors = [139, 237, 326, 456]
list_labels = ['city', 'signups', 'visitors', 'weekday']
list_cols = [cities, signups, visitors, weekdays]
zipped = list(zip(list_labels, list_cols))
df = dict(zipped)
users = pd.DataFrame(df)

##################################################################################################
"""Writing DataFrames"""

df.to_csv('output.csv')

##################################################################################################
"""Plotting DataFrames"""
""" These are wrapper over matplotlib so we can use it interchangebly"""


df.hist() #for histogram 
df.plot(x='xaxis',y='yaxis',kind='scatter',bins=30, range=(4,8), normed=True) #kind='box' kind='hist' 
plt.show()


from pandas.plotting import bootstrap_plot
bootstrap_plot(df, size=50, samples=500, color='grey')

##################################################################################################
"""Time Series"""

"""ISO 8601 format :: yyyy-mm-dd hh:mm:ss """

#parse dates 
df = pd.read_csv('df.csv', parse_dates=True, index_col= 'Date')

#Selecting single datetime
df.loc['2015-02-19 11:00:00']
#Selecting whole day
df.loc['2015-2-5']
#Slicing using dates/times
df.loc['2015-2-16':'2015-2-20']

#Filling missing values 
df.reindex(evening_2_11, method='ffill') #method='bfill'

daily_mean = df.resample('D').mean()

##################################################################################################
