"""
English-only plotting configuration
For systems without Chinese font support
"""

import matplotlib.pyplot as plt
import warnings

def setup_plotting():
    """Set up English-only plotting configuration"""

    # Reset matplotlib to default settings
    plt.rcdefaults()

    # Basic font configuration with fallback fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Arial', 'DejaVu Sans', 'Liberation Sans',
        'Bitstream Vera Sans', 'sans-serif'
    ]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    # Ensure text rendering
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    # Style configuration - use a safer style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')

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