import pandas as pd
import numpy as np
import seaborn as scs
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn import linear_model

data = pd.read_csv(r'C:\Users\hp\AppData\Local\Temp\7zO864F92FD\owid-covid-data.csv')

print(data.info)
print(data.head)


print(data.columns.values)
# Too many column headders #
# to do list #
# Graph data cases over time#
# drop irrelevant data smoking status, 'cardiovasc_death_rate', 'cardiovasc_death_rate' 'stringency_index' #

print(data.date)
pd.set_option('display.max_rows', 500)
print(data['location'].value_counts())

data.drop(['female_smokers','cardiovasc_death_rate', 'cardiovasc_death_rate' ,'stringency_index','diabetes_prevalence','handwashing_facilities','male_smokers','aged_65_older' ,'aged_70_older'], axis='columns', inplace=True)

print(data.shape)












