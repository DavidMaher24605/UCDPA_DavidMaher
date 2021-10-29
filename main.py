import pandas as pd
import numpy as np
import seaborn as scs
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
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

print(data.shape)

data.drop(['female_smokers','cardiovasc_death_rate', 'cardiovasc_death_rate' ,'stringency_index','diabetes_prevalence','handwashing_facilities','male_smokers','aged_65_older' ,'aged_70_older','excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
 'excess_mortality' ,'excess_mortality_cumulative_per_million'], axis='columns', inplace=True)

print(data.shape)
print(data['date'].min())
print(data['new_cases'].min())

print(data.loc[45]['date'])
print(data.loc[1361]['date'])
print(data.loc[4498]['date'])
print(data.loc[1]['new_cases'])
data['new_cases'] = data['new_cases'].fillna(0)

pos_count, neg_count = 0, 0

for num in data['new_cases']:

 # checking condition
 if num < 0:
  neg_count += 1

print("Negative numbers in the list: ", neg_count)

#merge date with other data set based on ISO code#
#Drop data with non nueric values#

#group data for ireland, chart data for Ireland, create function basedon if rate for a country is greater than a given number#


data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True)
Newcase_date = data['date']
Number_of_NewCases = data['new_cases']
plt.plot_date(Newcase_date, Number_of_NewCases, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.title('New cases per day')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.savefig('New Cases' + '.png')
plt.show()

#need to find out where cases are int#
grouped = data.groupby(data.continent)
Europe = grouped.get_group("Europe")
print(Europe)

Europe['date'] = pd.to_datetime(data['date'])
Europe.sort_values('date', inplace=True)
Newcase_date = Europe['date']
Number_of_NewCases = Europe['new_cases']
plt.plot_date(Newcase_date, Number_of_NewCases, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.title('New cases per day')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.savefig('New Cases' + '.png')
plt.show()

pos_count, neg_count = 0, 0

for num in Europe['new_cases']:

 # checking condition
 if num < 0:
  neg_count += 1

print("Negative numbers in the list: ", neg_count)

print('neg_count')

print()

Europe.new_cases =pd.to_numeric(Europe.new_cases, errors ='coerce').fillna(0).astype(int)

print(Europe[Europe['new_cases']<0])

Europe_NC = Europe.where(Europe['new_cases'] > 0, 0)

print(Europe_NC)
Europe_NC['date'] = pd.to_datetime(data['date'])
Europe_NC.sort_values('date', inplace=True)
Newcase_date = Europe_NC['date']
Number_of_NewCases = Europe_NC['new_cases']
plt.plot_date(Newcase_date, Number_of_NewCases, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.title('New cases per day')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.savefig('New Cases' + '.png')
plt.show()

# need to smooth out the data#
#need to constrain the dates#
#machine Leanring Linear regeression#

