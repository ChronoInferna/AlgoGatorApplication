from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
from scipy.stats import skew

# Step 1: Fetch data
tickers = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "VGK",
    "EWJ",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "BND",
    "DBC",
    "USO",
    "GLD",
    "SLV",
    "DBA",
    "UNG",
    "COPX",
    "UUP",
    "FXE",
    "FXY",
    "VWO",
]

start_date = datetime(year=2015, month=3, day=1)
end_date = datetime(year=2025, month=3, day=1)
data = (
    pd.DataFrame(
        yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
    )
    .dropna()
    .copy()
)
data = data.loc[:, data.columns.get_level_values(1) == "Close"]
returns = data.pct_change().dropna()

# Step 2: Parameters
lookbacks = [63, 126, 252]
vol_ewma_lambda = 0.94
annual_factor = 252
cost_per_unit = 0.0005  # Example transaction cost per turnover unit

# Step 3: Setup storage for results
results = []
vol = returns.ewm(alpha=1 - vol_ewma_lambda).std()

fig_pnl, ax_pnl = plt.subplots()
fig_dd, ax_dd = plt.subplots()
fig_asset_short, ax_asset_short = plt.subplots()
fig_asset_medium, ax_asset_medium = plt.subplots()
fig_asset_long, ax_asset_long = plt.subplots()

# Step 4: Loop over lookback periods (i.e. backtest)
for lb in lookbacks:
    # Signals
    signals = pd.DataFrame(index=returns.index, columns=returns.columns)
    for asset in returns.columns:
        # Actual backtesting loop
        for t in range(lb, len(data) - 1):
            y = np.log(data[asset].iloc[t - lb : t])
            x = np.arange(lb)
            X = add_constant(x)
            res = OLS(y, X).fit()
            signals.at[returns.index[t], asset] = res.params.iloc[1]

    signals = signals.astype(float)

    # Volatility scaling
    weights = signals / vol
    weights = weights.div(weights.abs().sum(axis=1), axis=0)

    # Turnover and transaction costs
    turnover = weights.diff().abs().sum(axis=1)
    cost = turnover * cost_per_unit

    # Portfolio PnL
    pnl = (weights.shift(1) * returns).sum(axis=1) - cost
    cum_pnl = (1 + pnl).cumprod()

    # Metrics
    mean_return = pnl.mean() * annual_factor
    volatility = pnl.std() * np.sqrt(annual_factor)
    sharpe = mean_return / volatility
    cum_max = cum_pnl.cummax()
    drawdown = (cum_pnl - cum_max) / cum_max
    max_dd = drawdown.min()
    skewness = skew(pnl.dropna())

    results.append(
        {
            "Lookback (days)": lb,
            "Annualized Return": round(mean_return, 4),
            "Annualized Volatility": round(volatility, 4),
            "Sharpe Ratio": round(sharpe, 4),
            "Max Drawdown": round(max_dd, 4),
            "Skew": round(skewness, 4),
            "Avg Turnover": round(turnover.mean(), 4),
        }
    )

    # Plot
    ax_pnl.plot(cum_pnl, label=f"Lookback {lb}d")
    ax_dd.plot(drawdown, label=f"Lookback {lb}d")
    contrib = weights.shift(1) * returns
    asset_axes = {63: ax_asset_short, 126: ax_asset_medium, 252: ax_asset_long}
    for asset in returns.columns:
        if lb in asset_axes:
            asset_axes[lb].plot(contrib[asset].cumsum(), label=str(asset[0]))

# Step 5: Results table
results_df = pd.DataFrame(results)
print(results_df)

# Step 6: Save plots
plot_parameters = [
    (
        "Cumulative PnL by Lookback Horizon",
        "Cumulative Return",
        "cumulative_pnl.png",
        ax_pnl,
        fig_pnl,
    ),
    ("Drawdowns by Lookback Horizon", "Drawdown", "drawdowns.png", ax_dd, fig_dd),
    (
        "Cumulative PnL Contribution by Asset (Short Horizon)",
        "Cumulative Return",
        "asset_contributions_short.png",
        ax_asset_short,
        fig_asset_short,
    ),
    (
        "Cumulative PnL Contribution by Asset (Medium Horizon)",
        "Cumulative Return",
        "asset_contributions_medium.png",
        ax_asset_medium,
        fig_asset_medium,
    ),
    (
        "Cumulative PnL Contribution by Asset (Long Horizon)",
        "Cumulative Return",
        "asset_contributions_long.png",
        ax_asset_long,
        fig_asset_long,
    ),
]

for title, ylabel, filename, ax, fig in plot_parameters:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(ncol=3, fontsize="small")
    fig.figure.savefig(filename)

print("Figures saved...")
