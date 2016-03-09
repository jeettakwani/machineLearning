# Imports

# pandas
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

business = pd.read_csv("../csv/business.csv")
review = pd.read_csv("../csv/review.csv",parse_dates=['date'],low_memory=False)
checkin  = pd.read_csv("../csv/checkin.csv")

#train.head(10)

#print train.head(10)

#fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
#sns.countplot(x='Open',hue='DayOfWeek', data=review,palette="hls", ax=axis1)

#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



# Date

# Create Year and Month columns
'''
train['Year']  = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))

test['Year']  = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))

# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)

# this column will be useful in analysis and visualization
train['Date'] = train['Date'].apply(lambda x: (str(x)[:7]))
test['Date']     = test['Date'].apply(lambda x: (str(x)[:7]))
'''

business_id = 'mVHrayjG3uZ_RLHkLj-AMg'
review_data = review[review["business_id"] == business_id]



# group by date and get average sales, and precent change
average_sales    = review_data.groupby('month')["stars"].sum()
pct_change_sales = review_data.groupby('month')["stars"].sum().pct_change()

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(15,8))

# plot average sales over time(year-month)
ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Sales store 1")
ax1.set_xticks(range(len(average_sales)))
ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)

plt.show()