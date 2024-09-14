import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from EFF import efficientFrontier as eff

path = "C:/Users/LENOVO/Desktop/FINA 4380/python 1/"
data = pd.read_excel(path + 'biggestETFData.xlsx', sheet_name='US-only', index_col=0)
data1 = pd.read_excel(path + 'biggestETFData.xlsx', sheet_name='international', index_col=0)
data2 = pd.read_excel(path + 'biggestETFData.xlsx', sheet_name='crossAsset', index_col=0)
data_m = data.resample('M').last()
data1_m = data1.resample('M').last()
data2_m = data2.resample('M').last()

data1_m = pd.concat([data_m, data1_m], axis=1)
data2_m = pd.concat([data1_m, data2_m], axis=1)

riskMeasure = 'CVaR'

frontier = eff(data_m, riskMeasure)
print(frontier)
frontier1 = eff(data1_m, riskMeasure)
frontier2 = eff(data2_m, riskMeasure)


plt.plot(frontier.volRange, frontier.muRange, c='r')
plt.show()