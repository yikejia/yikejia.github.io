import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#VIX Index

VIX = pd.read_csv('C:/Users/LENOVO/Desktop/Factor/^VIX.csv')
VIX =VIX.loc[:,['Date','Adj Close']]
VIX.set_index('Date', inplace=True)



data_spx = pd.read_csv('C:/Users/LENOVO/Desktop/Factor/SPX.csv')
data_spx1 = data_spx.loc[:,['Date','Adj Close']]
data_spx1.set_index('Date', inplace=True)
data_spx1_return = data_spx1.pct_change().dropna(how='all')
data_spx1_log = np.log(data_spx1/data_spx1.shift(1)).dropna()

New_total_VIX = pd.concat([VIX, data_spx1_log], axis=1, join="inner")
ic_5 = New_total_VIX.iloc[:,0].rolling(window=5, min_periods=1).corr(New_total_VIX.iloc[:,1])
ic_14 = New_total_VIX.iloc[:,0].rolling(window=14, min_periods=1).corr(New_total_VIX.iloc[:,1])
ic_21 = New_total_VIX.iloc[:,0].rolling(window=21, min_periods=1).corr(New_total_VIX.iloc[:,1])
ic_42 = New_total_VIX.iloc[:,0].rolling(window=42, min_periods=1).corr(New_total_VIX.iloc[:,1])
Date = New_total_VIX.index

ic_MEAN = [ic_5.mean(),ic_14.mean(),ic_21.mean(),ic_42.mean()]
ic_std = [ic_5.std(),ic_14.std(),ic_21.std(),ic_42.std()]
icIR = [ic_MEAN[x]/ic_std[x] for x in range(4)]
print(ic_MEAN)


plt.plot(Date, ic_5.cumsum(axis=0), marker='o',label = "ic_5")
plt.plot(Date, ic_14.cumsum(axis=0), marker='o',label = "ic_14")
plt.plot(Date, ic_21.cumsum(axis=0), marker='o',label = "ic_21")
plt.plot(Date, ic_42.cumsum(axis=0), marker='o',label = "ic_42")
plt.legend()
plt.show()




#Interest Rate Skewness
Interest= pd.read_csv('C:/Users/LENOVO/Desktop/Factor/DGS10.csv')
Interest =Interest.loc[:,['DATE','DGS10']]
Interest.set_index('DATE', inplace=True)
Interest.dropna(inplace = True)
def skewness(x):
    return np.mean(x-np.mean(x)**3)/np.power(np.mean(np.square(x)),3/2)
ic_MEAN_2 =[]
ic_std_2 =[]

for i in [5,14,21,42]:
    skewness_Interest = Interest.rolling(window= 14, min_periods=1).apply(skewness)
    New_total_IR = pd.concat([skewness_Interest, data_spx1_log], axis=1, join="inner")
    Date = New_total_IR.index
    ic = New_total_IR.iloc[:,0].rolling(window=i, min_periods=1).corr(New_total_IR.iloc[:,1])
    ic_MEAN_2 +=[ic.mean()]
    ic_std_2 +=[ic.std()]
    plt.plot(Date, ic.cumsum(axis=0), marker='o', label= "ic_" + str(i))

icIR_2 = [ic_MEAN_2[x]/ic_std_2[x] for x in range(4)]
print(icIR_2)
plt.legend()
plt.show()