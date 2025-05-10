import pandas as pd
import matplotlib.pyplot as plt

# Load the results CSV
df = pd.read_csv('reports/strategy_timeframe_comparison.csv')

# Filter for ETH-USD, Week timeframe
filtered = df[(df['symbol'] == 'ETH-USD') & (df['timeframe'] == 'Week')]

if not filtered.empty:
    ax = filtered.set_index('strategy')['total_return_%'].plot(
        kind='bar', legend=False, figsize=(8, 4), color='#1f77b4', edgecolor='black')
    plt.title('Strategy Comparison: ETH-USD (Week)')
    plt.ylabel('Total Return (%)')
    plt.xlabel('Strategy')
    plt.tight_layout()
    plt.savefig('docs/strategy_comparison.png')
    plt.close()
else:
    print('No data available for ETH-USD (Week) to plot.')
