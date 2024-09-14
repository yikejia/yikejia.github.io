import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

data_path = r"D:\course_ppt\Business school\FINA4380\factor_analysis\HW2.xlsx"

market_df = pd.read_csv(r"D:\course_ppt\Business school\FINA4380\factor_analysis\data\SPX.csv", index_col= 'Date')
stocks_df = pd.read_excel(r"D:\course_ppt\Business school\FINA4380\factor_analysis\data\Data.xlsx",index_col = 'DATE')
stocks_df = stocks_df.apply(pd.to_numeric, errors='coerce')


market_returns_df = market_df['Adj Close'].pct_change().dropna(how = "all")
market_returns_df.index = pd.to_datetime(market_returns_df.index)
market_returns_df.name = 'r_f'
stock_returns_df = stocks_df.pct_change().dropna(how = "all")
stock_returns_df.index = pd.to_datetime(stock_returns_df.index)
stock_returns_df.name = 'r'



window_years = 1
corr_window_years = 5
min_vol_days = 1
min_corr_days = 1

def calculate_beta(stock_returns, market_returns):
    stock_volatility = stock_returns.rolling(window=14, min_periods=min_vol_days).std()
    market_volatility = market_returns.rolling(window=14, min_periods=min_vol_days).std()
    correlation = market_returns.rolling(window=14, min_periods=1).corr(stock_returns)
    multiplier= stock_volatility.div(market_volatility, axis='index')
    beta_ts = correlation * multiplier
    # beta_ts = correlation * (stock_volatility / market_volatility)
    return 0.6 * beta_ts.fillna(0) + 0.4

merged_df = pd.concat([stock_returns_df, market_returns_df],axis = 1)
merged_df = merged_df.replace([np.inf, -np.inf], 0)
beta = calculate_beta(stock_returns_df, market_returns_df)


# beta_df = beta_df.apply(assign_to_portfolio)
def assign_to_portfolio(group):
    median_beta = group['beta'].median()
    group['portfolio'] = np.where(group['beta'] < median_beta, 'Low', 'High')
    return group

# merged_df = merged_df.apply(assign_to_portfolio)
median_beta = beta.median()
portfolio = np.where(beta < median_beta, 'Low', 'High')


def calculate_weights(group):
    beta_rank = group.rank()
    mean_rank = beta_rank.mean()
    k = 2 / abs(beta_rank - mean_rank).sum()
    weight_H = np.maximum(0, k * (beta_rank - mean_rank))
    weight_L = np.maximum(0, k * (mean_rank - beta_rank))
    return weight_H, weight_L

weight_H,weight_L = calculate_weights(beta)

def portfolio_returns(merged_df,weight_L,weight_H):
    stocks_df = merged_df.iloc[:, :-1]
    r_L = stocks_df * weight_L
    r_H = stocks_df * weight_H
    return r_L,r_H

r_L,r_H = portfolio_returns(merged_df,weight_L,weight_H)

def bab_returns(merged_df,beta,weight_L,weight_H,r_H,r_L):
    beta_L = beta * weight_L
    # if beta_L == 0: beta_L = beta
    beta_H = beta * weight_H
    # if beta_H == 0: beta_H = group['beta']
    # r_L = r_L.sum()
    # r_H = group['r_H'].sum()
    r_f = merged_df['r_f']
    part_L = (r_L.subtract(r_f, axis='index') / beta_L).replace([np.inf, -np.inf], 0)
    part_H = (r_H.subtract(r_f, axis='index') / beta_H).replace([np.inf, -np.inf], 0)
    r_BAB =  (part_L - part_H).dropna(how = "all")
    return r_BAB

r_BAB = bab_returns(merged_df,beta,weight_L,weight_H,r_H,r_L)
# print(merged_df)

IC = r_BAB.rolling(window=14, min_periods=1).corr(merged_df.iloc[:, :-1])
IC_MEAN = IC.mean()
IC_std = IC.std()
ICIR = IC_MEAN/IC_std
print(IC, ICIR)

cumulative_IC = IC.cumsum()

for column in cumulative_IC.columns:
    plt.plot(cumulative_IC.index, cumulative_IC[column], marker='o', label=column)
plt.legend()
plt.title("IC")
plt.xlabel("Date")
plt.ylabel("IC Value")

# 旋转日期标签以避免重叠
plt.xticks(rotation=45)

# 显示图形
plt.show()



# merged_df['Date'] = pd.to_datetime(merged_df['Date'])
# if not merged_df.empty:
#     for year, group in merged_df.groupby(merged_df['Date'].dt.year):
#         group[['Date', 'r_BAB']].drop_duplicates().dropna().to_csv(data_path + f'BAB_{year}.csv', index=False)
#
# merged_bab_data = pd.DataFrame()
# for file in os.listdir(data_path):
#     if file.startswith('BAB_') and file.endswith('.csv'):
#         df = pd.read_csv(os.path.join(data_path, file), usecols=['Date', 'r_BAB'])
#         merged_bab_data = pd.concat([merged_bab_data, df])
#
# merged_bab_data.drop_duplicates().dropna().to_csv(data_path + 'BAB_only.csv', index=False)