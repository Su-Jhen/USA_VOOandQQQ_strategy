"""
English-only plotting configuration
For systems without Chinese font support
"""

import matplotlib.pyplot as plt
import warnings

def setup_plotting():
    """Set up English-only plotting configuration"""

    # Basic font configuration
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    # Style configuration
    plt.style.use('seaborn-v0_8-darkgrid')

    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # Suppress font warnings
    warnings.filterwarnings('ignore', message='Glyph.*missing from current font')
    warnings.filterwarnings('ignore', message='Font.*not found')

def get_english_labels():
    """Get English labels mapping"""
    return {
        # Chart titles
        'backtest_results': 'MA Crossover Strategy Backtest Results',
        'equity_curve': 'Equity Curve Comparison',
        'returns_comparison': 'Cumulative Returns Comparison',
        'allocation_changes': 'Asset Allocation Changes',
        'drawdown_analysis': 'Strategy Drawdown Analysis',
        'market_states': 'Market State Distribution',
        'monthly_returns': 'Monthly Returns Distribution',

        # Axis labels
        'date': 'Date',
        'value': 'Asset Value ($)',
        'returns_pct': 'Returns (%)',
        'allocation_pct': 'Allocation (%)',
        'drawdown_pct': 'Drawdown (%)',
        'frequency': 'Frequency',
        'monthly_returns': 'Monthly Returns (%)',

        # Legend labels
        'ma_strategy': 'MA Strategy',
        'voo_buy_hold': 'VOO Buy & Hold',
        'qqq_buy_hold': 'QQQ Buy & Hold',
        'voo': 'VOO',
        'qqq': 'QQQ',
        'bull': 'Bull Market',
        'bear': 'Bear Market',
        'neutral': 'Neutral',
        'average': 'Average'
    }

# Initialize plotting setup
setup_plotting()