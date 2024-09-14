from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from Functions import *


# The csv contains Russell 2000 constituents
Russell2000 = ' '.join(pd.read_excel('Russell-2000.xlsx')['Ticker'].to_list())
period = 125

'''
data = yf.download(Russell2000, period="5y")
with open("RU2000_data.pickle", "wb") as f:
    pickle.dump(data, f)
'''

# Russell 2000
data = pickle.load(open("RU2000_data.pickle", "rb"))
RU_close = data.xs('Adj Close', axis=1)
RU_close.interpolate(method="spline", order=3, inplace=True)
RU_close = RU_close.ffill()
RU_close.dropna(how='any', axis=1, inplace=True)
RU_return = (np.log(RU_close/RU_close.shift(1)))

# Remove first row of all data for return calculation
RU_close = RU_close.iloc[1:,]
RU_return = RU_return.iloc[1:,]

Composite_score = []
Momentum_list = []
Mean_reversion_list = []
for t in tqdm(range(period, len(RU_return), period)):
    RU_return_t = RU_return.iloc[t-period:t, :]
    RU_close_t = RU_close.iloc[t-period:t, :]

    '''Momentum score'''
    Momentum = RU_return_t.iloc[-period:-period*11//12, :].sum()
    Momentum = (Momentum - Momentum.mean()) / Momentum.std()
    Momentum_list.append(Momentum)

    '''Mean reversion score, fitting parameters for an OU model'''
    Aux_process = []
    for i in range(1, period):
        Aux_process.append(np.sum(RU_return_t[-period:-period+i], axis=0))
    Aux_process = pd.DataFrame(Aux_process)
    Aux_process_lag = (Aux_process.shift()).iloc[1:]
    Aux_process = Aux_process.iloc[1:]

    b = []
    zeta = []
    for stock in Aux_process.columns:
        Xn = sm.add_constant(Aux_process_lag.loc[:, stock])
        Xn1 = Aux_process[stock]
        res = sm.OLS(Xn1, Xn).fit()
        b.append(res.params)
        zeta.append(res.resid)
    zeta = pd.concat(zeta, axis=1)

    b = pd.DataFrame(b)
    b1 = np.diag(b.iloc[:, 1:])
    b = pd.DataFrame(b['const'])
    b['beta'] = b1
    m = b['const']/(1-b['beta'])
    b.loc[b['beta'] >= 1, 'beta'] = 0.99
    Var_zeta = zeta.var()
    sigma_eq = np.sqrt(np.divide(Var_zeta, (1-np.square(b['beta']))))

    # The fitted S-score, times negative one, since in the paper, strategy was to sell when it is high
    Mean_reversion = np.divide((np.array(RU_return_t[-period:-1].sum())-m), sigma_eq)
    Mean_reversion.replace([np.inf, -np.inf], np.nan, inplace=True)
    Mean_reversion = (Mean_reversion - Mean_reversion.mean()) * -1 / Mean_reversion.std()
    Mean_reversion_list.append(Mean_reversion)

    # Combining the two for composite score
    Composite_score.append((np.array(Momentum)+Mean_reversion)/2)


Composite_score = pd.concat(Composite_score, axis=1).T
Momentum_list = pd.concat(Momentum_list, axis=1).T
Mean_reversion_list = pd.concat(Mean_reversion_list, axis=1).T
Composite_score.columns = RU_return.columns
Momentum_list.columns = RU_return.columns
Mean_reversion_list.columns = RU_return.columns

# IC score calculation
period_return = calculate_period_returns(RU_close, period)

ic_list = []
ic_momentum = []
ic_reversion = []

for i in range(len(Composite_score.index)-1):
    ic_list.append(Composite_score.iloc[i].corr(period_return.iloc[i+1], method='spearman'))
    ic_momentum.append(Momentum_list.iloc[i].corr(period_return.iloc[i + 1], method='spearman'))
    ic_reversion.append(Mean_reversion_list.iloc[i].corr(period_return.iloc[i+1], method='spearman'))

plt.plot(ic_list, label='combined')
plt.plot(ic_momentum, label='momentum')
plt.plot(ic_reversion, label='reversion')
plt.legend()
plt.show()

print('average ic: ', np.average(ic_list))
print('average ic momentum: ', np.average(ic_momentum))
print('average ic reversion: ', np.average(ic_reversion))
print('ir : ', np.average(ic_list)/np.std(ic_list))
print('momentum ir : ', np.average(ic_momentum)/np.std(ic_momentum))
print('reversion ir : ', np.average(ic_reversion)/np.std(ic_reversion))



