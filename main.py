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











