#!/usr/bin/env python3
"""
測試圖表中文字體顯示
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 添加src/main/python到路徑
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from utils.plot_config import apply_chinese_style, get_safe_title, create_bilingual_labels

def test_font_display():
    """測試中文字體顯示"""
    print("測試matplotlib中文字體顯示...")

    # 應用中文字體設置
    apply_chinese_style()
    labels = create_bilingual_labels()

    # 創建測試數據
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    values = np.cumsum(np.random.randn(len(dates))) + 1000

    # 創建測試圖表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(get_safe_title('MA交叉策略回測結果'), fontsize=14, fontweight='bold')

    # 測試1：線圖
    ax1 = axes[0, 0]
    ax1.plot(dates, values, label=labels['ma_strategy'])
    ax1.set_title(get_safe_title('權益曲線比較'))
    ax1.set_xlabel(labels['date'])
    ax1.set_ylabel(labels['value'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 測試2：柱狀圖
    ax2 = axes[0, 1]
    categories = ['策略A', '策略B', '策略C']
    safe_categories = [get_safe_title(cat) for cat in categories]
    returns = [15.2, 12.8, 18.5]
    ax2.bar(safe_categories, returns)
    ax2.set_title(get_safe_title('策略報酬比較'))
    ax2.set_ylabel(labels['returns_pct'])

    # 測試3：餅圖
    ax3 = axes[1, 0]
    state_data = [60, 30, 10]
    state_labels = ['Bull', 'Bear', 'Neutral']
    ax3.pie(state_data, labels=state_labels, autopct='%1.1f%%')
    ax3.set_title(get_safe_title('市場狀態分布'))

    # 測試4：散點圖
    ax4 = axes[1, 1]
    x = np.random.randn(50)
    y = np.random.randn(50)
    ax4.scatter(x, y, alpha=0.6)
    ax4.set_title(get_safe_title('風險報酬散布'))
    ax4.set_xlabel('風險 / Risk')
    ax4.set_ylabel('報酬 / Return')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 儲存測試圖表
    output_file = 'output/font_test_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 測試圖表已儲存: {output_file}")
    print("請檢查圖表中的中文是否正確顯示")

    # 顯示字體信息
    current_font = plt.rcParams['font.family']
    print(f"當前使用字體: {current_font}")

def main():
    """主函數"""
    try:
        test_font_display()
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()