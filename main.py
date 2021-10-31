import pandas as pd
import numpy as np
import seaborn as sns
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

data.drop(['female_smokers','cardiovasc_death_rate','cardiovasc_death_rate','stringency_index','diabetes_prevalence','handwashing_facilities','male_smokers','aged_65_older' ,'aged_70_older','excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
 'excess_mortality','excess_mortality_cumulative_per_million'], axis='columns', inplace=True)

print(data.shape)
print(data['date'].min())
print(data['new_cases'].min())

print(data.loc[45]['date'])
print(data.loc[1361]['date'])
print(data.loc[4498]['date'])
print(data.loc[1]['new_cases'])
data['new_cases'] = data['new_cases'].fillna(0)
data['location'] = data['location'].fillna(0)
data['location'].replace('', np.nan, inplace=True)
data['location'] = data['location'].astype(str)
data['date'] = data['date'].astype('datetime64[ns]')
data['date'] = pd.to_datetime(data['date'], format='%y%m%d')

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
Europe_group_sort= Europe.sort_values("location").head()
print(Europe_group_sort['location'])

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

 # checking condition#
 if num < 0:
  neg_count += 1

print("Negative numbers in the list: ", neg_count)

print('neg_count')

Europe.new_cases =pd.to_numeric(Europe.new_cases, errors ='coerce').fillna(0).astype(int)
Europe['location'] = Europe['location'].replace("0","unknown")
#Europe["location"] = pd.to_string(Europe["location"])#
print(Europe[Europe['new_cases']<0])

Europe_NC = Europe.where(Europe['new_cases'] > 0, 0)

print(Europe_NC['location'].value_counts())

#finding pattern with Regex#
Regex_search =Europe_NC['location'].str.contains(r'0')

#sns.lineplot(data=Europe_NC, x="date", y="new_cases")#

print(Europe_NC.shape)

Europe_NC_EU= Europe_NC.drop(Europe_NC.index[Europe_NC['location'].isin([' ',0,'Andorra','United Kingdom','vatican','San Marino','Liechtenstein','Monaco','Iceland','Kosovo','Bosnia and Herzegovina','Switzerland','Montenegro','Belarus','Russia','Serbia','Ukraine','North Macedonia','Albania','Norway'])])
date_formatt= mpl_dates.DateFormatter('%d-%m-%Y')

#Europe_NC_2021 = Europe_NC.loc[(Europe_NC['date'] > '01-01-2021')]#
#Europe_NC['date'] = pd.to_datetime(Europe_NC['date']).dt.date#
print(Europe_NC_EU.location)






# repalce issue stopping nice chart not regression, must move on for now#



#need to fill 0 with blank
# need to smooth out the data#
#need to constrain the dates#
#machine Leanring Linear regeression#


# prepare the lists for the model
# X = date_format
# y = data['gravi_deceduti'].tolist()[1:]
# # date format is not suitable for modeling, let's transform the date into incrementals number starting from April 1st
# starting_date = 37  # April 1st is the 37th day of the series
# day_numbers = []
# for i in range(1, len(X)):
#     day_numbers.append([i])
# X = day_numbers
# # # let's train our model only with data after the peak
# X = X[starting_date:]
# y = y[starting_date:]
# # Instantiate Linear Regression
# linear_regr = linear_model.LinearRegression()
# # Train the model using the training sets
# linear_regr.fit(X, y)
# print ("Linear Regression Model Score: %s" % (linear_regr.score(X, y)))#