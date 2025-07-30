# main.py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import requests
import time

# --- Configuration ---
# IMPORTANT: You MUST get your own free API key from: https://www.alphavantage.co/support/#api-key
# Paste your key in the quotes below or enter it when the script runs.
ALPHA_VANTAGE_API_KEY = "4VU401PRTAD5KDVR"  # <-- PASTE YOUR KEY HERE

# Define the benchmark tickers for the Indian market.
BENCHMARK_TICKERS = ['^NSEI', '^BSESN']

# Define the analysis period to end on the last day of the previous month
today = datetime.date.today()
first_day_of_current_month = today.replace(day=1)
END_DATE = first_day_of_current_month - datetime.timedelta(days=1)
START_DATE = END_DATE - datetime.timedelta(days=365 * 2) # 2 years of data

# Define the risk-free rate for Sharpe Ratio calculation
RISK_FREE_RATE = 0.04

# --- Helper Functions ---

def get_api_key():
    """Gets the Alpha Vantage API key from the user if not already set."""
    global ALPHA_VANTAGE_API_KEY
    if not ALPHA_VANTAGE_API_KEY:
        ALPHA_VANTAGE_API_KEY = input("Please enter your personal Alpha Vantage API key: ").strip()
    if not ALPHA_VANTAGE_API_KEY:
        print("Warning: No API key provided. Fundamental data will be limited.")
    return ALPHA_VANTAGE_API_KEY

def get_user_portfolio():
    """
    Prompts the user to enter their stock portfolio details in a single line.
    """
    print("\n--- Enter Your Portfolio Details ---")
    print("Please enter your portfolio in a single line.")
    print("FORMAT: TICKER1 SHARES1, TICKER2 SHARES2, ...")
    print("EXAMPLE: RELIANCE 10, TCS 20, TATAMOTORS 5")
    
    portfolio_str = input("> ").strip()
    
    if not portfolio_str:
        print("\nNo stocks were entered. Exiting analysis.")
        return None
        
    portfolio = {}
    entries = [entry.strip() for entry in portfolio_str.split(',')]
    
    for entry in entries:
        parts = entry.split()
        if len(parts) != 2:
            print(f"Warning: Skipping invalid entry '{entry}'. Please use 'TICKER SHARES' format.")
            continue
            
        ticker, shares_str = parts
        ticker = ticker.upper()
        
        if not ticker.endswith(('.NS', '.BO')):
            ticker = f"{ticker}.NS"

        try:
            shares = float(shares_str)
            if shares <= 0:
                print(f"Warning: Skipping '{ticker}' due to invalid shares count '{shares_str}'.")
                continue
            portfolio[ticker] = shares
        except ValueError:
            print(f"Warning: Skipping '{ticker}' due to invalid shares format '{shares_str}'.")
            continue
            
    if not portfolio:
        print("\nNo valid stocks could be read from your input. Exiting analysis.")
        return None

    print("\n--- Your Final Portfolio ---")
    for ticker, shares in portfolio.items():
        print(f"- {ticker}: {shares} shares")
    print("-" * 28)
    return portfolio

def fetch_fundamental_data(ticker, api_key):
    """
    Fetches fundamental data for a single ticker.
    Tries Alpha Vantage first, then falls back to yfinance for sector info.
    """
    details = {'Sector': 'Unknown'}

    # --- Primary Source: Alpha Vantage (for P/E, EPS, etc.) ---
    if api_key:
        symbol_to_try = ticker.replace('.NS', '').replace('.BO', '')
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol_to_try}&apikey={api_key}'
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            
            if "Note" in data:
                print(f"-> Info: Alpha Vantage API limit likely reached for {ticker}.")
            elif "PERatio" in data and data.get("PERatio") not in [None, "None", "0"]:
                details = {
                    'P/E Ratio': float(data.get('PERatio', 0)),
                    'Market Cap': f"₹{int(data.get('MarketCapitalization', 0)) / 10**7:,.2f} Cr",
                    'EPS': float(data.get('EPS', 0)),
                    'Dividend Yield': float(data.get('DividendYield', 0)) * 100,
                    'Sector': data.get('Sector', 'Unknown')
                }
        except Exception as e:
            print(f"-> Info: Could not fetch from Alpha Vantage for {ticker}. Error: {e}")

    # --- Fallback Source: yfinance (especially for Sector) ---
    if details['Sector'] == 'Unknown':
        print(f"-> Info: Trying yfinance as a fallback for sector info for {ticker}...")
        try:
            stock_info = yf.Ticker(ticker).info
            if 'sector' in stock_info and stock_info['sector']:
                details['Sector'] = stock_info['sector']
                print(f"   Success: Found sector '{details['Sector']}' via yfinance.")
            else:
                print(f"   Info: yfinance also does not have sector data for {ticker}.")
        except Exception:
            print(f"   Warning: Could not fetch detailed info for {ticker} from yfinance.")
            
    return details


def download_data(tickers, start_date, end_date):
    """Downloads historical closing prices for a list of tickers."""
    print(f"\nDownloading price data for: {', '.join(tickers)}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, group_by='ticker')
        if data.empty:
            return None
        
        close_prices = pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                close_prices[ticker] = data[ticker]['Close']

        return close_prices.dropna(axis=1, how='all')
    except Exception as e:
        print(f"An unexpected error occurred during price data download: {e}")
        return None

def analyze_and_plot_allocation(portfolio, all_prices, fundamental_data):
    """Calculates and plots the portfolio's sector allocation."""
    print("\n--- Portfolio Allocation Analysis ---")
    
    latest_prices = all_prices.iloc[-1]
    allocation = {}
    for ticker, shares in portfolio.items():
        if ticker not in latest_prices.index or pd.isna(latest_prices[ticker]):
            continue
        
        current_value = latest_prices[ticker] * shares
        sector = fundamental_data.get(ticker, {}).get('Sector', 'Unknown')
        
        allocation[sector] = allocation.get(sector, 0) + current_value
            
    if not allocation or all(v == 'Unknown' for v in allocation.keys()):
        print("Could not determine portfolio allocation from any source.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    # Create a new figure for this plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        allocation.values(), 
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=90,
        pctdistance=0.85
    )
    
    ax.legend(wedges, allocation.keys(), title="Sectors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title("Portfolio Sector Allocation", fontsize=16, fontweight='bold')
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')
    plt.tight_layout()
    print("Generating Sector Allocation pie chart...")
    # We do NOT call plt.show() here anymore.

def analyze_contributors_and_detractors(portfolio, all_prices):
    """Analyzes which stocks contributed most to gains and losses."""
    print("\n--- Performance Contribution Analysis ---")
    
    start_prices = all_prices.iloc[0]
    end_prices = all_prices.iloc[-1]
    
    contributions = {}
    for ticker, shares in portfolio.items():
        if ticker not in start_prices.index or ticker not in end_prices.index or pd.isna(start_prices[ticker]) or pd.isna(end_prices[ticker]):
            continue
        
        contribution = (end_prices[ticker] - start_prices[ticker]) * shares
        contributions[ticker] = contribution
        
    if not contributions:
        print("Could not analyze contributions due to missing price data.")
        return
        
    sorted_contributions = sorted(contributions.items(), key=lambda item: item[1])
    total_change = sum(contributions.values())
    
    print(f"Over the period, your portfolio's value changed by: ₹{total_change:,.2f}")
    print("-" * 50)
    print(f"{'Ticker':<15} {'Contribution (₹)':<20}")
    print("-" * 50)
    for ticker, value in sorted_contributions:
        print(f"{ticker:<15} {value:,.2f}")
    print("-" * 50)
    
    if sorted_contributions:
        print(f"Top Contributor (Winner): {sorted_contributions[-1][0]} (+₹{sorted_contributions[-1][1]:,.2f})")
        print(f"Top Detractor (Loser):  {sorted_contributions[0][0]} (₹{sorted_contributions[0][1]:,.2f})")
    print("-" * 50)

def calculate_performance_metrics(returns_series):
    """Calculates key performance metrics from a series of daily returns."""
    if returns_series.empty: return {}
        
    total_return = (returns_series + 1).prod() - 1
    annualized_volatility = returns_series.std() * np.sqrt(252)
    
    if annualized_volatility == 0: return {'Total Return': total_return, 'Annualized Volatility': 0, 'Annualized Sharpe Ratio': float('inf')}

    excess_returns = returns_series - (RISK_FREE_RATE / 252)
    annualized_sharpe_ratio = (excess_returns.mean() * 252) / annualized_volatility
    
    return {
        'Total Return': total_return,
        'Annualized Volatility': annualized_volatility,
        'Annualized Sharpe Ratio': annualized_sharpe_ratio
    }

def plot_performance(cumulative_portfolio_returns, cumulative_benchmarks_returns):
    """Plots the cumulative performance of the portfolio against multiple benchmarks."""
    plt.style.use('seaborn-v0_8-darkgrid')
    # Create a new figure for this plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(cumulative_portfolio_returns.index, cumulative_portfolio_returns, label='My Portfolio', color='royalblue', linewidth=2.5, zorder=10)
    
    colors = ['gray', 'orangered']
    linestyles = ['--', ':']
    for i, (ticker, returns) in enumerate(cumulative_benchmarks_returns.items()):
        ax.plot(returns.index, returns, label=f'{ticker}', color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=2)
    
    ax.set_title('Portfolio Performance vs. Benchmarks', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns', fontsize=12)
    ax.legend(fontsize=12, title='Legend')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    print("\nGenerating Performance vs. Benchmark chart...")
    # We do NOT call plt.show() here anymore.

# --- Main Execution ---

def main():
    """Main function to run the portfolio analysis."""
    portfolio = get_user_portfolio()
    if not portfolio: return

    api_key = get_api_key()
    fundamental_data = {}
    
    print("\nFetching fundamental data...")
    print("(This may take a moment due to API rate limits)...")
    for i, ticker in enumerate(portfolio.keys()):
        print(f"Fetching data for {ticker}...")
        data = fetch_fundamental_data(ticker, api_key)
        if data: fundamental_data[ticker] = data
        # We still sleep even if using yfinance to be considerate
        if api_key and i < len(portfolio.keys()) - 1:
            time.sleep(15)

    all_tickers = list(portfolio.keys()) + BENCHMARK_TICKERS
    all_prices = download_data(all_tickers, START_DATE, END_DATE)
    
    if all_prices is None or all_prices.empty:
        print("\nCould not retrieve price data for any assets. Exiting analysis.")
        return

    portfolio_prices = all_prices.drop(columns=BENCHMARK_TICKERS, errors='ignore')
    valid_tickers = [ticker for ticker in portfolio.keys() if ticker in portfolio_prices.columns and not portfolio_prices[ticker].isnull().all()]
    missing_tickers = set(portfolio.keys()) - set(valid_tickers)

    if missing_tickers:
        print("\n--- IMPORTANT WARNING ---")
        print(f"Could not download price data for the following tickers: {', '.join(missing_tickers)}")
        print("They will be EXCLUDED from all calculations.\n")

    if not valid_tickers:
        print("Error: Could not retrieve data for ANY of the stocks in your portfolio. Exiting.")
        return

    valid_portfolio = {ticker: portfolio[ticker] for ticker in valid_tickers}
    
    # --- Run Analyses and Prepare Plots ---
    allocation_plotted = False
    if any(d.get('Sector', 'Unknown') != 'Unknown' for d in fundamental_data.values()):
        analyze_and_plot_allocation(valid_portfolio, portfolio_prices, fundamental_data)
        allocation_plotted = True
    else:
        print("\nSkipping Sector Allocation chart: No sector data could be retrieved from any source for the provided stocks.")
    
    analyze_contributors_and_detractors(valid_portfolio, portfolio_prices)
    
    # --- Existing Analysis Continues ---
    if any('P/E Ratio' in d for d in fundamental_data.values()):
        print("\n--- Fundamental Analysis ---")
        header = f"{'Ticker':<15} {'P/E Ratio':<15} {'Market Cap':<20} {'EPS':<15} {'Dividend Yield (%)':<20}"
        print(header)
        print("-" * len(header))
        for ticker, data in fundamental_data.items():
            if ticker in valid_portfolio and 'P/E Ratio' in data:
                print(f"{ticker:<15} {data.get('P/E Ratio', 'N/A'):<15.2f} {data.get('Market Cap', 'N/A'):<20} {data.get('EPS', 'N/A'):<15.2f} {data.get('Dividend Yield', 'N/A'):<20.2f}")
        print("-" * len(header))
        
    portfolio_value = (portfolio_prices[list(valid_portfolio.keys())] * pd.Series(valid_portfolio)).sum(axis=1).dropna()
    
    if portfolio_value.empty:
        print("Could not calculate portfolio value. Exiting.")
        return

    portfolio_returns = portfolio_value.pct_change().dropna()
    portfolio_metrics = calculate_performance_metrics(portfolio_returns)
    cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1

    benchmark_prices = all_prices.drop(columns=list(portfolio.keys()), errors='ignore')
    benchmark_metrics_dict = {}
    cumulative_benchmarks_returns_dict = {}
    for ticker in BENCHMARK_TICKERS:
        if ticker in benchmark_prices.columns:
            returns = benchmark_prices[ticker].pct_change().dropna()
            if not returns.empty:
                benchmark_metrics_dict[ticker] = calculate_performance_metrics(returns)
                cumulative_benchmarks_returns_dict[ticker] = (1 + returns).cumprod() - 1

    print("\n--- Performance Analysis Results ---")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}\n")
    
    header = f"{'Metric':<25} {'My Portfolio':<20}"
    for ticker in benchmark_metrics_dict: header += f" {ticker:<20}"
    print(header)
    print("-" * len(header))
    
    for metric_name in ['Total Return', 'Annualized Volatility', 'Annualized Sharpe Ratio']:
        row = f"{metric_name:<25}"
        portfolio_val = portfolio_metrics.get(metric_name)
        if portfolio_val is not None:
            row += f" {portfolio_val:<20.2%}" if metric_name != 'Annualized Sharpe Ratio' else f" {portfolio_val:<20.2f}"
        else:
            row += f" {'N/A':<20}"
            
        for ticker in benchmark_metrics_dict:
            bench_val = benchmark_metrics_dict.get(ticker, {}).get(metric_name)
            if bench_val is not None:
                row += f" {bench_val:<20.2%}" if metric_name != 'Annualized Sharpe Ratio' else f" {bench_val:<20.2f}"
            else:
                row += f" {'N/A':<20}"
        print(row)
    print("-" * len(header))

    performance_plotted = False
    if not cumulative_portfolio_returns.empty:
        plot_performance(cumulative_portfolio_returns, cumulative_benchmarks_returns_dict)
        performance_plotted = True
    else:
        print("\n--- IMPORTANT ---")
        print("Could not generate Performance vs. Benchmark chart.")
        print("This usually happens if there is not enough historical price data for your selected stocks in the chosen date range.")

    # --- NEW: Show all generated plots at once at the very end ---
    if allocation_plotted or performance_plotted:
        print("\nDisplaying all generated charts...")
        plt.show()


if __name__ == '__main__':
    main()
