import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r'C:\Users\hp\AppData\Local\Temp\7zO864F92FD\owid-covid-data.csv')
Country_Flag_URLs = pd.read_csv(r'C:\Users\hp\Downloads\countries_continents_codes_flags_url.csv')

print(data.info)
print(data.head)

print(data.columns.values)
print(Country_Flag_URLs.columns.values)
# drop irrelevant data smoking status, 'cardiovasc_death_rate', 'cardiovasc_death_rate' 'stringency_index' #
print(data.date)
pd.set_option('display.max_rows', 500)
print(data['location'].value_counts())
print(data.shape)
data.drop(['female_smokers', 'cardiovasc_death_rate', 'cardiovasc_death_rate', 'stringency_index', 'diabetes_prevalence', 'handwashing_facilities', 'male_smokers', 'aged_65_older', 'aged_70_older', 'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
 'excess_mortality', 'excess_mortality_cumulative_per_million'], axis='columns', inplace=True)
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
data['new_tests'].replace('', np.nan, inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data['new_tests'] = data['new_tests'].fillna(0)
data['location'] = data['location'].astype(str)
data['date'] = data['date'].apply(lambda x: pd.Timestamp(x).strftime('%m-%d-%Y'))

pos_count, neg_count = 0, 0

for num in data['new_cases']:

# checking condition #

 if num < 0:
  neg_count += 1

print("Negative numbers in the list: ", neg_count)

# Merge data with other dataset based on ISO code#

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

# need to find out where cases are int #
Europe = data.loc[data['continent'] == "Europe"]
Europe_group_sort = Europe.sort_values("location").head()
print(Europe.shape)
print(Europe_group_sort['location'])

pos_count, neg_count = 0, 0

for num in Europe['new_cases']:

 # checking condition#
 if num < 0:
    neg_count += 1

print("Negative numbers in the list: ", neg_count)

print('neg_count')


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

Europe.new_cases =pd.to_numeric(Europe.new_cases, errors ='coerce').fillna(0).astype(int)
Europe['location'] = Europe['location'].replace("0","unknown")
print(Europe[Europe['new_cases']<0])

Europe_NC = Europe.where(Europe['new_cases'] > 0, 0)

print(Europe_NC['location'].value_counts())

# Finding pattern with Regex#
Regex_search = Europe_NC['location'].str.contains(r'0')

print(Europe_NC.shape)
Europe_NC_EU = Europe_NC.drop(Europe_NC.index[Europe_NC['location'].isin([' ', 0, 'Andorra', 'United Kingdom', 'vatican', 'San Marino', 'Liechtenstein', 'Monaco', 'Moldova', 'Iceland', 'Kosovo', 'Bosnia and Herzegovina', 'Switzerland', 'Montenegro', 'Belarus', 'Russia', 'Serbia', 'Ukraine', 'North Macedonia', 'Albania', 'Norway'])])
Europe_NC_EU['date'] = Europe_NC['date'].apply(lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))

print(Europe_NC_EU.location)

print(Europe_NC_EU.date)
print(Europe_NC_EU.date.max())
Europe_NC_EU['date'] = pd.to_datetime(Europe_NC_EU['date'])
Europe_NC_EU.sort_values('date', inplace=True)
Newcase_date = Europe_NC_EU['date']
Number_of_NewCases = Europe_NC_EU['new_cases']
plt.plot_date(Newcase_date, Number_of_NewCases, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.title('New cases per day')
plt.xlabel('Date')
plt.ylabel('New Cases 2021')
plt.savefig('New Cases EU 2021' + '.png')
plt.show()

Europe_NC_EU_sort_date = Europe_NC_EU.sort_values("date")

print(data['date'].max())
print(Europe_NC_EU.date.max())
print(Europe_NC_EU_sort_date['date'].max())

# repalce issue stopping nice chart not regression, must move on for now#

# data frames will be joined on an shared ISO code#

print(data['iso_code'])
print(Country_Flag_URLs['alpha-3'])
# Renaming column to make joing easier#
Country_Flag_URLs.rename(columns={'alpha-3': 'iso_code', }, inplace=True)
print(Country_Flag_URLs['iso_code'])

data_with_flagURls = pd.merge(data, Country_Flag_URLs, on="iso_code")
pd.options.display.max_colwidth = 100
print(data_with_flagURls['image_url'])

# Data frames merged successfully#

Europe_NC_EU_2021 = Europe_NC_EU.loc[(Europe_NC_EU['date'] >= '2021-01-01') & (Europe_NC_EU['date'] < '2021-10-23')]
print(Europe_NC_EU_2021)
print(Europe_NC_EU_2021.date)
print(Europe_NC_EU_2021.date.max())
Europe_NC_EU_2021['date'] = pd.to_datetime(Europe_NC_EU['date'])
Europe_NC_EU_2021.sort_values('date', inplace=True)
Newcase_date = Europe_NC_EU_2021['date']
Number_of_NewCases = Europe_NC_EU_2021['new_cases']
plt.plot_date(Newcase_date, Number_of_NewCases, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.title('New cases per day')
plt.xlabel('Date')
plt.ylabel('New Cases 2021')
plt.savefig('New Cases EU 2021' + '.png')
plt.show()

Europe_NC_EU_2021['date'] = pd.to_datetime(Europe_NC_EU['date'])
Europe_NC_EU_2021.sort_values('date', inplace=True)
Newcase_date = Europe_NC_EU_2021['date']
Number_of_NewCases = Europe_NC_EU_2021['new_cases_smoothed']
plt.plot_date(Newcase_date, Number_of_NewCases, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.title('New cases per day')
plt.xlabel('Date')
plt.ylabel('New Cases 2021')
plt.savefig('New Cases EU 2021' + '.png')
plt.show()

print(Europe_NC_EU_2021.head)

Europe_NC_EU_2021.columns.to_list().index('iso_code')
Europe_NC_EU_2021['new_tests'] = Europe_NC_EU_2021['new_tests'].fillna(0)
print(Europe_NC_EU_2021.new_tests.min())
print(Europe_NC_EU_2021.new_tests.max())

# alpha-3 from one data set has the same three letter acronym as the iso_code column for OWID data set#

sns.boxplot(y='location', x='new_cases', data=Europe_NC_EU_2021)
plt.title('Box Plot of New Cases in The EU by Country - Non Standardised')
plt.show()

sns.boxplot(y='continent', x='new_cases', data=Europe_NC_EU_2021)
plt.show()

sns.boxplot(y='continent', x='new_tests', data=Europe_NC_EU_2021)
plt.show()

# box plots reveal that there are a significant number of outliers#
# data needs to be normalised to get further insights e.g. regression#
fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(Europe_NC_EU_2021['date'], Europe_NC_EU_2021['new_cases'])
ax.set_xlabel('date')
ax.set_ylabel('new_cases')
plt.title('New Cases per Day 2021 - Non Standardised')
plt.show()

Europe_NC_EU_2021['zscore-new_cases'] = zscore(Europe_NC_EU_2021['new_cases'])
Europe_NC_EU_2021['zscore-new_tests'] = zscore(Europe_NC_EU_2021['new_tests'])
Europe_NC_EU_2021['zscore-total_deaths'] = zscore(Europe_NC_EU_2021['total_deaths'])
Europe_NC_EU_2021['zscore-people_fully_vaccinated'] = zscore(Europe_NC_EU_2021['people_fully_vaccinated'])

threshold = 3
Europe_NC_EU_2021['outliers_new_cases'] = np.where((Europe_NC_EU_2021['zscore-new_cases'] - threshold > 0), True, np.where(Europe_NC_EU_2021['zscore-new_cases'] + threshold < 0, True, False))
Europe_NC_EU_2021['outliers_new_tests'] = np.where((Europe_NC_EU_2021['zscore-new_tests'] - threshold > 0), True, np.where(Europe_NC_EU_2021['zscore-new_tests'] + threshold < 0, True, False))
Europe_NC_EU_2021['outliers_total_deaths'] = np.where((Europe_NC_EU_2021['zscore-total_deaths'] - threshold > 0), True, np.where(Europe_NC_EU_2021['zscore-total_deaths'] + threshold < 0, True, False))
Europe_NC_EU_2021['outliers_people_fully_vaccinated'] = np.where((Europe_NC_EU_2021['zscore-people_fully_vaccinated'] - threshold > 0), True, np.where(Europe_NC_EU_2021['zscore-people_fully_vaccinated'] + threshold < 0, True, False))

Europe_NC_EU_2021.drop(Europe_NC_EU_2021[Europe_NC_EU_2021['outliers_new_cases'] == True].index, inplace=True)
Europe_NC_EU_2021.drop(Europe_NC_EU_2021[Europe_NC_EU_2021['outliers_new_tests'] == True].index, inplace=True)
Europe_NC_EU_2021.drop(Europe_NC_EU_2021[Europe_NC_EU_2021['outliers_total_deaths'] == True].index, inplace=True)
Europe_NC_EU_2021.drop(Europe_NC_EU_2021[Europe_NC_EU_2021['outliers_people_fully_vaccinated'] == True].index, inplace=True)

# lets run that box plot again#
fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(Europe_NC_EU_2021['date'], Europe_NC_EU_2021['new_cases'])
ax.set_xlabel('date')
ax.set_ylabel('new_cases')
plt.title('New Cases per Day 2021 - Within 3 Standard Deviations')
plt.show()

sns.boxplot(y='location', x='new_cases', data=Europe_NC_EU_2021)
plt.title('Box Plot of New Cases in The EU by Country - Standardised')
plt.show()

# a compaison of the scatter plots shows a more even distribution with less outliers#


# deciding on the algorythm https://towardsdatascience.com/choosing-a-scikit-learn-linear-regression-algorithm-dd96b48105f5#

# need to smooth out the data#
# machine learning Linear regression#

# X = Europe_NC_EU_2021.iloc[:,27].values.reshape(-1, 1)  # values converts it into a numpy array
# Y = Europe_NC_EU_2021.iloc[:,5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()
# linear_regressor.fit(X, Y)
# Y_pred = linear_regressor.predict(X)
# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='red')
# plt.show()

print(Europe_NC_EU_2021.isnull().any())
Europe_NC_EU_2021 = Europe_NC_EU_2021.fillna(lambda x: x.median())
print(Europe_NC_EU_2021.isnull().any())

Europe_NC_EU_2021['new_cases'] = pd.to_numeric(Europe_NC_EU_2021['new_cases'], errors='coerce')
Europe_NC_EU_2021['new_tests'] = pd.to_numeric(Europe_NC_EU_2021['new_tests'], errors='coerce')


# X = Europe_NC_EU_2021.iloc[:,27].values.reshape(-1, 1)  # values converts it into a numpy array
# Y = Europe_NC_EU_2021.iloc[:,5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
# linear_regressor = LinearRegression()
# linear_regressor.fit(X, Y)
# Y_pred = linear_regressor.predict(X)
# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='red')
# plt.show()

Europe_NC_EU_2021_IRL_FR_DE = Europe_NC_EU_2021.drop(Europe_NC_EU_2021.index[Europe_NC_EU_2021['location'].isin(['Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'Greece', 'Hungary'])])
sns.lineplot(x="date", y="new_cases", hue="location", data=Europe_NC_EU_2021_IRL_FR_DE)
plt.title('New Cases per Day 2021')
plt.show()

