# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

business = pd.read_csv("../csv/train_business.csv")
review = pd.read_csv("../csv/train_review.csv",parse_dates=['date'],low_memory=False)
checkin  = pd.read_csv("../csv/train_checkin.csv")

review_test = pd.read_csv("../csv/test_review.csv",parse_dates=['date'],low_memory=False)
review_test_groupby = DataFrame({'count':review_test.groupby(['year','month'])['stars'].count()})


review_me=DataFrame({'count':review.groupby(['year','month'])['stars'].count()})

review_test_mean = [ review_test_groupby['count'].mean() for x in range(134)]
print review_test_mean


rms = sqrt(mean_squared_error(review_test_groupby['count'].tolist(), review_test_mean))
print rms


dates = pd.date_range('2005-03-14', '2015-12-24')


plt.plot(review_me['count'].tolist(),'-ro')
plt.title("Review count")
plt.xlabel("Year")
plt.ylabel("Review Count")
plt.xticks([w*12 for w in range(11)],
  [w for w in range(2005,2016)])
plt.autoscale(tight=True)
plt.grid()

#plt.show()

#AO = Series(s[:,1], index=dates)
#print AO