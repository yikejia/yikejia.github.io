import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
path = "C:/Users/LENOVO/Desktop/FINA 4380/python 1/biggestETFData.xlsx"
data = pd.read_excel(path, sheet_name='US-only', index_col=0)
data_m = data.resample('M').last()
data_m_ret = data_m.pct_change().dropna(how='all')
ER = data_m_ret.mean()
omega = data_m_ret.cov()
muRange = np.arange(0.005, 0.014, 0.0002)
volRange = np.zeros(len(muRange))
n = len(data_m_ret.columns)
weight = pd.DataFrame()
lb = 0.0
ub = 1.0
bnd = ((lb, ub),)

riskMeasure = 'MV' #VaR, CVaR
def MV(w, mat):
    return np.dot(w.T, np.dot(mat, w))

for j in range(1, n):
    bnd = bnd + ((lb, ub),)

for i in range(len(muRange)):
    mu = muRange[i]
    x_0 = np.ones(n)/n
    consTR = ({'type': 'eq', 'fun': lambda x: 1- np.sum(x)},
              {'type': 'eq', 'fun': lambda x: mu-np.dot(x, ER)})
    w = minimize(MV, x_0, args=(omega), method='SLSQP', bounds=bnd, constraints=consTR)
    volRange[i] = MV(w.x, omega)
    temp = pd.DataFrame(w.x, index=data.columns, columns=[mu])
    weight = pd.concat([weight, temp], axis=1)

plt.plot(volRange, muRange, c='r')
plt.show()



