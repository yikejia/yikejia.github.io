from matplotlib import pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pickle
from scipy.stats import spearmanr
from Functions import *


# The csv contains stocks that are S&P500 constituents
SP500 = pd.read_csv('constituents.csv')['Symbol'].to_list()
SP500str = ' '.join(pd.read_csv('constituents.csv')['Symbol'].to_list())

period = 126

'''SPY = yf.download("SPY", period="10y")
with open("SPY.pickle", "wb") as f:
    pickle.dump(SPY, f)'''

# SPY (Market proxy)
SPY = pickle.load(open("SPY.pickle", "rb"))
SPY_close = SPY.xs('Adj Close', axis=1)
SPY_return = (np.log(SPY_close / SPY_close.shift(1)))
SPY_return.interpolate(method="spline", order=3, inplace=True)
SPY_return = SPY_return.fillna(0)

'''data = yf.download(SP500str, period="10y")
with open("SP500_data.pickle", "wb") as f:
    pickle.dump(data, f)'''

# S&P 500
data = pickle.load(open("SP500_data.pickle", "rb"))
SP_close = data.xs('Adj Close', axis=1)
SP_volume = data.xs('Volume', axis=1)
SP_close.dropna(how='all', axis=1, inplace=True)
SP_return = (np.log(SP_close/SP_close.shift(1)))
SP_return.interpolate(method="spline", order=3, inplace=True)
SP_return = SP_return.fillna(0)

# Risk-free rate
RF_rate = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv', index_col=0)['RF']
RF_rate.index = pd.to_datetime(RF_rate.index, format='%Y%m%d')
RF_rate = SP_return.merge(RF_rate, how='left', left_index=True, right_index=True).iloc[:, -1]
RF_rate.bfill(inplace=True)
RF_rate.ffill(inplace=True)

# Remove first row of all data for return calculation
SP_close = SP_close.iloc[1:,]
SP_return = SP_return.iloc[1:,]
SP_volume = SP_volume.iloc[1:,]
SPY_close = SPY_close.iloc[1:,]
SPY_return = SPY_return.iloc[1:,]
RF_rate = RF_rate.iloc[1:, ]

# Initialize all the features
idio_vol = []
coskew = []
idio_skew = []
market_cap = []
lagged_monthly_return = []
momentum = []
max_return = []
price_impact = []
turnover = []
skew = []
beta = []
beta2 = []

# Loop through the 10-year period
for t in tqdm(range(period, len(SP_return), period)):
    SP_return_t = SP_return.iloc[t-period:t, :]
    SP_close_t = SP_close.iloc[t-period:t, :]
    SP_volume_t = SP_volume.iloc[t - period:t, :]

    '''Coding the features'''
    # CAPM
    resid, alpha, beta_row = CAPM(SP_return_t, SPY_return, RF_rate)

    # CAPM2
    resid2, alpha2, beta2_row = CAPM2(SP_return_t, SPY_return, RF_rate)

    # Idiosyncratic volatility
    idio_vol_row = resid.std()
    idio_vol.append(idio_vol_row)

    # Coskewness
    coskew_row = SP_return_t.apply(lambda column: (SPY_return**2).cov(column))
    coskew.append(coskew_row)

    # Idiosyncratic skewness
    idio_skew_row = (((resid2 - resid2.mean())/resid2.std()) ** 3).mean()
    idio_skew.append(idio_skew_row)

    # beta
    beta.append(beta_row)

    # beta for M squared
    beta2.append(beta2_row)

    # Lagged monthly return, probably no need use shift
    lagged_monthly_return_row = ((SP_close_t.shift(period//12) - SP_close_t.shift(period//6)) /
                                 SP_close_t.shift(period//12)).iloc[-1]
    lagged_monthly_return.append(lagged_monthly_return_row)

    # Momentum
    momentum_row = ((SP_close_t.shift(period//6) - SP_close_t.shift(period-1)) / SP_close_t.shift(period-1)).iloc[-1]
    momentum.append(momentum_row)

    # Maximum return
    max_return_row = calculate_max_return(SP_return_t.index[-1], SP_return_t)
    max_return.append(max_return_row)

    # Price impact
    price_impact_row = (SP_return_t.iloc[:, :].abs()/SP_volume_t.iloc[:, :]).mean()
    price_impact.append(price_impact_row)

    # Skewness (Target)
    skew_row = (((SP_return_t - SP_return_t.mean())/SP_return_t.std()) ** 3).mean()
    skew.append(skew_row)

# Rank among stock for every features
beta = pd.concat(beta, axis=0).T.rank().T/474
beta2 = pd.concat(beta2, axis=0).T.rank().T/474
idio_vol = pd.DataFrame(idio_vol).T.rank().T/474
coskew = pd.DataFrame(coskew).T.rank().T/474
idio_skew = pd.DataFrame(idio_skew).T.rank().T/474
market_cap = pd.DataFrame(market_cap).T.rank().T/474
lagged_monthly_return = pd.DataFrame(lagged_monthly_return).T.rank().T/474
momentum = pd.DataFrame(momentum).T.rank().T/474
max_return = pd.DataFrame(max_return).T.rank().T/474
price_impact = pd.DataFrame(price_impact).T.rank().T/474
turnover = pd.DataFrame(turnover).T.rank().T/474
skew = pd.DataFrame(skew).T.rank().T/474
previous_skew = skew.shift(1)
dataframe_list = [idio_vol, coskew, idio_skew, lagged_monthly_return, momentum, max_return, price_impact, previous_skew]

# Calculate the return over each period
period_return = calculate_period_returns(SP_close, period)

# Generate a list of dataframes, each dataframe corresponding to one period, 474 stocks' features
X_list = combine_df_rg(dataframe_list)


ic_list_skew = []
ic_list_coskew = []
ic_list_return = []
# Run regression on relationship between T-1 features and T skewness.
# Then, use that model to predict T+1 skewness with features at T
for i in range(1, len(X_list)-2):
    X_list[i].fillna(X_list[i].mean(), inplace=True)
    X = sm.add_constant(X_list[i], prepend=True)
    y1 = skew.iloc[i+1].values
    res = sm.OLS(y1, X, missing='drop').fit()
    X1 = sm.add_constant(X_list[i + 1], prepend=True)

    pred_y2 = res.predict(X1)
    y2 = skew.iloc[i+2].values
    ic_skew = spearmanr(y2, pred_y2, nan_policy='omit').correlation
    ic_list_skew.append(ic_skew)

    return2 = period_return.iloc[i+2]
    ic_return = spearmanr(pred_y2, return2, nan_policy='omit').correlation
    ic_list_return.append(ic_return)

print('average skew ic:', np.average(ic_list_skew))
print('skew ir:', np.average(ic_list_skew)/np.std(ic_list_skew))

print('average return ic:', np.average(ic_list_return))
print('return ir:', np.average(ic_list_return)/np.std(ic_list_return))

plt.plot(ic_list_skew, label='IC with skewness')
plt.plot(ic_list_return, label='IC with return')
plt.legend()
plt.show()
