#!/usr/bin/env python3
"""
簡化版測試腳本
驗證增強功能是否正常運作
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.ma_crossover import create_ma_parameter_matrix
from backtest_ma_strategy import MAStrategyBacktest, MAStrategy

def test_enhanced_features():
    """測試增強功能"""
    print("🧪 測試增強版MA策略功能")
    print("="*50)

    # 1. 測試參數矩陣創建
    print("\n1️⃣ 測試參數矩陣創建...")
    strategies = create_ma_parameter_matrix()
    print(f"   ✅ 成功創建 {len(strategies)} 組參數")

    # 顯示前5組參數
    for i, (name, params) in enumerate(list(strategies.items())[:5], 1):
        print(f"   {i}. {name}: {params.fast_period}/{params.slow_period}MA, QQQ權重={params.qqq_weight_bear*100:.0f}%")

    # 2. 測試進階優化參數
    print("\n2️⃣ 測試進階優化參數...")
    advanced_strategy = MAStrategy(
        fast_period=50,
        slow_period=200,
        qqq_weight_bear=0.5,
        use_slope_confirm=True,
        use_crossover_filter=True,
        crossover_threshold=0.02,
        use_duration_confirm=True,
        confirm_days=3,
        use_dynamic_weight=True
    )
    print(f"   ✅ 進階策略參數: {advanced_strategy}")

    # 3. 測試快速回測
    print("\n3️⃣ 執行快速回測測試...")
    try:
        backtest = MAStrategyBacktest(
            strategy_params=advanced_strategy,
            start_date='2020-01-01',  # 縮短期間
            end_date='2022-01-01',
            initial_capital=10000,
            annual_contribution=1000
        )

        results = backtest.run_backtest(show_progress=False)
        metrics = results['metrics']

        print(f"   ✅ 回測成功完成")
        print(f"   📊 CAGR: {metrics['cagr']:.2f}%")
        print(f"   📊 夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"   📊 最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"   📊 交易次數: {metrics['num_trades']}")

    except Exception as e:
        print(f"   ❌ 回測失敗: {str(e)}")
        return False

    print("\n🎉 所有功能測試通過！")
    return True

if __name__ == "__main__":
    test_enhanced_features()