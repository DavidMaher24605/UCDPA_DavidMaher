import pandas as pd
import numpy as np
import seaborn as scs
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

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

data.del(columns=['female_smokers'], axis=1)

print(data.columns.values)










