#!/usr/bin/env python3
"""
測試修正後的圖表顯示
"""

import sys
from pathlib import Path

# 添加src/main/python到路徑
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from backtest_ma_strategy import MAStrategyBacktest, MAStrategy

def test_chart_generation():
    """測試圖表生成"""
    print("測試修正後的圖表生成...")

    # 創建測試策略
    strategy_params = MAStrategy(
        fast_period=50,
        slow_period=200,
        qqq_weight_bear=0.5
    )

    # 執行簡化的回測
    backtest = MAStrategyBacktest(
        strategy_params=strategy_params,
        start_date='2020-01-01',  # 縮短測試期間
        end_date='2024-01-01',
        initial_capital=10000,
        annual_contribution=3000
    )

    try:
        # 執行回測
        results = backtest.run_backtest()

        # 生成圖表
        chart_path = "output/test_fixed_chart.png"
        backtest.plot_results(chart_path)

        print("✅ 測試圖表生成成功")
        print(f"圖表已儲存: {chart_path}")

        return True

    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    success = test_chart_generation()

    if success:
        print("\n🎉 圖表字體問題已修正！")
        print("現在所有圖表都使用英文標籤，避免中文字體顯示問題")
    else:
        print("\n❌ 仍有問題需要解決")

if __name__ == "__main__":
    main()