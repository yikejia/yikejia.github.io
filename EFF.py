import pandas as pd
import numpy as np
from scipy.optimize import minimize

class efficientFrontier:
    def __init__(self, prices, riskMeasure):
        self.prices = prices
        self.riskMeasure = riskMeasure
        self.alpha = 0.01
        self.lb = 0.0
        self.ub = 1.0
        self.n = len(prices.columns)
        self.muRange = np.arange(0.005, 0.014, 0.0002)
        self.ret = prices.pct_change().dropna()
        self.ER = self.ret.mean()
        self.omega = self.ret.cov()

        self.volRange = self.frontier()['volRange']
        self.optimWeight = self.frontier()['weight']

    def MV(self, w, mat):
        return np.dot(w.T, np.dot(mat, w))

    def VaR(self, w, R):
        return abs(np.percentile(np.dot(R, w), self.alpha))

    def CVaR(self, w, R):
        portRet = np.dot(R, w)
        return abs(np.average(    portRet[portRet<np.percentile(np.dot(R, w), self.alpha)]     ))

    def frontier(self):
        wgt = {}
        volRange = np.zeros(len(self.muRange))
        bnd = ((self.lb, self.ub),)
        for j in range(1, self.n):
            bnd = bnd + ((self.lb, self.ub),)
        for i in range(len(self.muRange)):
            mu = self.muRange[i]
            wgt[mu] = []
            x_0 = np.ones(self.n) / self.n
            consTR = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)},
                      {'type': 'eq', 'fun': lambda x: mu - np.dot(x, self.ER)})
            if self.riskMeasure == 'MV':
                w = minimize(self.MV, x_0, args=(self.omega), method='SLSQP', bounds=bnd, constraints=consTR)
                volRange[i] = self.MV(w.x, self.omega)
            elif self.riskMeasure == 'VaR':
                w = minimize(self.VaR, x_0, args=(self.ret), method='SLSQP', bounds=bnd, constraints=consTR)
                volRange[i] = self.VaR(w.x, self.ret)
            else:
                w = minimize(self.CVaR, x_0, args=(self.ret), method='SLSQP', bounds=bnd, constraints=consTR)
                volRange[i] = self.CVaR(w.x, self.ret)

            wgt[mu].extend(np.squeeze(w.x))
        wgtVec = pd.DataFrame.from_dict(wgt,orient='columns').transpose()

        return {'volRange': volRange, 'weight': wgtVec}