"""
繪圖配置模組
解決matplotlib中文字體顯示問題
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings

def setup_chinese_fonts():
    """設置中文字體顯示"""

    # 獲取系統類型
    system = platform.system()

    # 常見中文字體列表
    chinese_fonts = [
        'SimHei',           # 黑体 (Windows)
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'PingFang SC',      # 苹方 (macOS)
        'Hiragino Sans GB', # 冬青黑体 (macOS)
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'Noto Sans CJK SC', # 思源黑体 (Linux)
        'DejaVu Sans',      # 後備字體
    ]

    # 嘗試找到可用的中文字體
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None

    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break

    # 設置字體
    if chinese_font:
        plt.rcParams['font.family'] = [chinese_font, 'sans-serif']
        print(f"✅ 使用中文字體: {chinese_font}")
    else:
        # 如果找不到中文字體，嘗試使用系統默認
        if system == 'Windows':
            plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['PingFang SC', 'Hiragino Sans GB', 'sans-serif']
        else:  # Linux
            plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']

        print("⚠️ 未找到首選中文字體，使用系統默認")

    # 其他字體配置
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    # 抑制字體警告
    warnings.filterwarnings('ignore', message='Glyph.*missing from current font')
    warnings.filterwarnings('ignore', message='Font.*not found')

def get_safe_title(title: str) -> str:
    """
    獲取安全的標題（如果字體不支持中文則使用英文）

    Args:
        title: 原始標題

    Returns:
        安全的標題字符串
    """
    # 中英文標題對照表
    title_mapping = {
        'MA交叉策略回測結果': 'MA Crossover Strategy Backtest Results',
        '權益曲線比較': 'Equity Curve Comparison',
        '累積報酬率': 'Cumulative Returns',
        '資產配置變化': 'Asset Allocation Changes',
        '策略回撤': 'Strategy Drawdown',
        '市場狀態分布': 'Market State Distribution',
        '月度報酬分布': 'Monthly Returns Distribution',
        '資產價值': 'Asset Value',
        '報酬率': 'Returns (%)',
        '配置比例': 'Allocation (%)',
        '回撤': 'Drawdown (%)',
        '日期': 'Date',
        '頻率': 'Frequency',
        '月報酬率': 'Monthly Returns (%)',
        'MA策略': 'MA Strategy',
        '平均': 'Average'
    }

    return title_mapping.get(title, title)

def apply_chinese_style():
    """應用中文圖表樣式"""
    setup_chinese_fonts()

    # 設置圖表樣式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 顏色配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def create_bilingual_labels():
    """創建雙語標籤映射"""
    return {
        # 圖表標題
        'backtest_results': '回測結果 / Backtest Results',
        'equity_curve': '權益曲線 / Equity Curve',
        'returns_comparison': '報酬比較 / Returns Comparison',
        'allocation_changes': '配置變化 / Allocation Changes',
        'drawdown_analysis': '回撤分析 / Drawdown Analysis',
        'market_states': '市場狀態 / Market States',
        'monthly_returns': '月度報酬 / Monthly Returns',

        # 軸標籤
        'date': '日期 / Date',
        'value': '價值 ($) / Value ($)',
        'returns_pct': '報酬率 (%) / Returns (%)',
        'allocation_pct': '配置比例 (%) / Allocation (%)',
        'drawdown_pct': '回撤 (%) / Drawdown (%)',
        'frequency': '頻率 / Frequency',

        # 圖例
        'ma_strategy': 'MA策略 / MA Strategy',
        'voo_buy_hold': 'VOO買入持有 / VOO Buy&Hold',
        'qqq_buy_hold': 'QQQ買入持有 / QQQ Buy&Hold',
        'voo': 'VOO',
        'qqq': 'QQQ',
        'bull': '牛市 / Bull',
        'bear': '熊市 / Bear',
        'neutral': '中性 / Neutral',
        'average': '平均 / Average'
    }

# 初始化字體設置（導入時自動執行）
try:
    setup_chinese_fonts()
except Exception as e:
    print(f"字體設置警告: {e}")
    print("將使用默認字體，可能無法正確顯示中文")